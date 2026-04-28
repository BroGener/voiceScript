"""
transcriber_whisperx.py
=======================
WhisperX transcriber with Route B diarization.

Route B flow:
  ASR → word alignment → pyannote segmentation (boundary only)
  → embedding extraction per segment
  → constrained clustering against known speaker centroids
  → post-processing (merge gaps, remove short segments)
  → word assignment

Cold-start flow (empty DB):
  ASR → word alignment → DiarizationPipeline (full black-box)
  → try matching against DB (may be partly populated)
  → write pending_<stem>.json for user to name unknowns
  → on next run, Route B takes over

whisperx 3.x DiarizationPipeline returns a DataFrame, not a pyannote Annotation.
The conversion must handle both cases.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from config import cfg
from diarization_postprocessor import DiarizationPostProcessor
from utils import Segment, assign_seg_ids, fmt_srt


class WhisperXTranscriber:

    def __init__(self) -> None:
        self._asr_model     = None
        self._align_model   = None
        self._align_meta    = None
        self._seg_model     = None
        self._embed_model   = None
        self._diarize_model = None
        self._cfg = cfg.whisperx
        self._dev = cfg.device

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path | str,
        speaker_manager=None,
    ) -> list[Segment]:
        audio_path = Path(audio_path)
        print(f"\n[WhisperX] Transcribing: {audio_path.name}")

        raw     = self._run_asr(audio_path)
        aligned = self._run_align(raw["segments"], audio_path)

        has_speakers = (speaker_manager is not None
                        and speaker_manager.has_speakers())

        if has_speakers:
            segments = self._route_b_diarize(audio_path, aligned, speaker_manager)
        else:
            segments = self._coldstart_diarize(audio_path, aligned, speaker_manager)

        # Assign stable seg_ids
        assign_seg_ids(segments, audio_path.stem)
        print(f"[WhisperX] Done: {len(segments)} segments.")
        return segments

    def unload(self) -> None:
        for attr in ("_asr_model", "_align_model", "_seg_model",
                     "_embed_model", "_diarize_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self._align_meta = None
        gc.collect()
        try:
            import torch
            if self._dev.device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("[WhisperX] Models unloaded.")

    # ------------------------------------------------------------------
    # Route B: constrained clustering
    # ------------------------------------------------------------------

    def _route_b_diarize(self, audio_path, aligned, speaker_manager) -> list[Segment]:
        print("[WhisperX] Route B: segmentation + constrained clustering ...")

        raw_segments = self._run_segmentation(audio_path)
        if not raw_segments:
            print("[WhisperX] Segmentation returned nothing — falling back.")
            return self._coldstart_diarize(audio_path, aligned, speaker_manager)

        embeddings, valid_segs = self._extract_segment_embeddings(audio_path, raw_segments)
        if not embeddings:
            print("[WhisperX] No embeddings — falling back.")
            return self._coldstart_diarize(audio_path, aligned, speaker_manager)

        speaker_labels = self._constrained_cluster(embeddings, speaker_manager)

        diarize_records = [
            {"speaker": lbl, "start": seg["start"], "end": seg["end"]}
            for lbl, seg in zip(speaker_labels, valid_segs)
        ]

        post = DiarizationPostProcessor()
        diarize_records = post.process_records(diarize_records)

        segments = self._assign_words(aligned, diarize_records)
        return self._to_segments(segments)

    def _run_segmentation(self, audio_path: Path) -> list[dict]:
        if self._seg_model is None:
            from pyannote.audio import Model
            print("[WhisperX] Loading segmentation model ...")
            self._seg_model = Model.from_pretrained(
                "pyannote/segmentation", use_auth_token=cfg.hf_token
            )
            print("[WhisperX] Segmentation model loaded.")

        from pyannote.audio import Inference
        inference = Inference(self._seg_model, window="sliding",
                              duration=5.0, step=0.5)
        try:
            output    = inference(str(audio_path))
            scores    = output.data.max(axis=1)
            frames    = output.sliding_window
            threshold = 0.5
            segments  = []
            in_speech = False
            seg_start = 0.0

            for i, score in enumerate(scores):
                t = frames[i].middle
                if score >= threshold and not in_speech:
                    seg_start = frames[i].start
                    in_speech = True
                elif score < threshold and in_speech:
                    if t - seg_start >= cfg.speaker.min_segment_duration_s:
                        segments.append({"start": seg_start, "end": t})
                    in_speech = False

            if in_speech:
                segments.append({"start": seg_start, "end": frames[-1].end})

            print(f"[WhisperX] Segmentation: {len(segments)} speech segments.")
            return segments
        except Exception as ex:
            print(f"[WhisperX] Segmentation error: {ex}")
            return []

    def _extract_segment_embeddings(self, audio_path, segments):
        if self._embed_model is None:
            from pyannote.audio import Model
            print("[WhisperX] Loading embedding model ...")
            self._embed_model = Model.from_pretrained(
                "pyannote/embedding", use_auth_token=cfg.hf_token
            )
            print("[WhisperX] Embedding model loaded.")

        from pyannote.audio import Inference
        import torchaudio

        inference = Inference(self._embed_model, window="whole")
        waveform, sr = torchaudio.load(str(audio_path))
        embeddings, valid_segs = [], []
        min_dur = cfg.speaker.min_segment_duration_s

        for seg in segments:
            if seg["end"] - seg["start"] < min_dur:
                continue
            s     = int(seg["start"] * sr)
            e     = int(seg["end"]   * sr)
            chunk = waveform[:, s:e]
            try:
                emb = inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr})
                embeddings.append(np.array(emb))
                valid_segs.append(seg)
            except Exception as ex:
                print(f"[WhisperX] Embedding failed {seg['start']:.1f}-{seg['end']:.1f}: {ex}")

        print(f"[WhisperX] {len(embeddings)} embeddings extracted.")
        return embeddings, valid_segs

    def _constrained_cluster(self, new_embeddings, speaker_manager) -> list[str]:
        min_conf = cfg.speaker.remap_min_confidence
        labels   = [""] * len(new_embeddings)
        unmatched_indices = []

        for i, emb in enumerate(new_embeddings):
            best_name, best_score = speaker_manager.match_speaker(emb)
            if best_name is not None and best_score >= min_conf:
                labels[i] = best_name
            else:
                unmatched_indices.append(i)

        if unmatched_indices:
            unmatched_vecs = np.stack([new_embeddings[i] for i in unmatched_indices])
            if len(unmatched_vecs) == 1:
                cluster_labels = [0]
            else:
                dist_matrix   = pdist(unmatched_vecs, metric="cosine")
                Z             = linkage(dist_matrix, method="average")
                n_new         = max(1, self._cfg.max_speakers - len(speaker_manager.all_centroids()))
                cluster_labels = fcluster(Z, t=n_new, criterion="maxclust").tolist()

            for idx, cid in zip(unmatched_indices, cluster_labels):
                labels[idx] = f"SPEAKER_NEW_{cid - 1:02d}"

        matched   = sum(1 for l in labels if not l.startswith("SPEAKER_NEW"))
        unmatched = len(labels) - matched
        print(f"[WhisperX] {matched} matched to known speakers, {unmatched} new/unknown.")
        return labels

    # ------------------------------------------------------------------
    # Cold-start: DiarizationPipeline fallback
    # ------------------------------------------------------------------

    def _coldstart_diarize(self, audio_path, aligned, speaker_manager) -> list[Segment]:
        print("[WhisperX] Cold-start: using DiarizationPipeline ...")

        if self._diarize_model is None:
            import whisperx
            print("[WhisperX] Loading DiarizationPipeline ...")
            self._diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=cfg.hf_token,
                device=self._dev.device,
            )
            print("[WhisperX] DiarizationPipeline loaded.")

        kwargs: dict = {}
        if self._cfg.num_speakers is not None:
            kwargs["num_speakers"] = self._cfg.num_speakers
        else:
            kwargs["min_speakers"] = self._cfg.min_speakers
            kwargs["max_speakers"] = self._cfg.max_speakers

        raw_result = self._diarize_model(str(audio_path), **kwargs)

        # whisperx 3.x returns a DataFrame; pyannote returns an Annotation.
        # Handle both.
        diarize_records = self._parse_diarize_result(raw_result)

        if not diarize_records:
            print("[WhisperX] ⚠️  DiarizationPipeline returned no segments.")
        else:
            print(f"[WhisperX] Got {len(diarize_records)} diarization records, "
                  f"speakers: {sorted({r['speaker'] for r in diarize_records})}")

        # Apply post-processing
        post = DiarizationPostProcessor()
        diarize_records = post.process_records(diarize_records)

        # Try matching against known speakers (partial DB)
        if speaker_manager is not None and speaker_manager.has_speakers():
            diarize_records = self._remap_coldstart_records(
                diarize_records, audio_path, speaker_manager
            )

        # Write pending mapping for unknowns
        self._write_pending_mapping(audio_path, diarize_records)

        segments = self._assign_words(aligned, diarize_records)
        return self._to_segments(segments)

    def _parse_diarize_result(self, raw_result) -> list[dict]:
        """
        Parse diarization output into [{speaker, start, end}] records.
        Handles both:
          - whisperx 3.x DataFrame (columns: segment, speaker, ...)
          - pyannote Annotation object
        """
        records = []

        # Case 1: pandas DataFrame (whisperx 3.x native format)
        try:
            import pandas as pd
            if isinstance(raw_result, pd.DataFrame):
                for _, row in raw_result.iterrows():
                    seg = row.get("segment", None)
                    if seg is not None:
                        records.append({
                            "speaker": str(row.get("label", row.get("speaker", "SPEAKER_00"))),
                            "start":   float(seg.start),
                            "end":     float(seg.end),
                        })
                    elif "start" in row and "end" in row:
                        records.append({
                            "speaker": str(row.get("speaker", "SPEAKER_00")),
                            "start":   float(row["start"]),
                            "end":     float(row["end"]),
                        })
                if records:
                    return records
        except Exception:
            pass

        # Case 2: pyannote Annotation
        try:
            for turn, _, speaker in raw_result.itertracks(yield_label=True):
                records.append({
                    "speaker": str(speaker),
                    "start":   float(turn.start),
                    "end":     float(turn.end),
                })
            if records:
                return records
        except AttributeError:
            pass

        # Case 3: whisperx assign_word_speakers result (dict with 'segments')
        try:
            if isinstance(raw_result, dict) and "segments" in raw_result:
                for seg in raw_result["segments"]:
                    if "speaker" in seg:
                        records.append({
                            "speaker": seg["speaker"],
                            "start":   float(seg["start"]),
                            "end":     float(seg["end"]),
                        })
                if records:
                    return records
        except Exception:
            pass

        print("[WhisperX] ⚠️  Could not parse diarization result "
              f"(type={type(raw_result).__name__})")
        return records

    def _remap_coldstart_records(
        self, records, audio_path, speaker_manager
    ) -> list[dict]:
        """
        After cold-start clustering, try matching temp labels to known speakers
        by computing a per-label centroid and comparing to the DB.
        Labels that match get renamed; unknowns keep their SPEAKER_XX name.
        """
        print("[WhisperX] Attempting DB remap of cold-start labels ...")

        # Group records by temp label
        label_records: dict[str, list[dict]] = {}
        for rec in records:
            label_records.setdefault(rec["speaker"], []).append(rec)

        # Extract per-label centroids
        try:
            from pyannote.audio import Model, Inference
            import torchaudio

            model     = Model.from_pretrained("pyannote/embedding",
                                              use_auth_token=cfg.hf_token)
            inference = Inference(model, window="whole")
            waveform, sr = torchaudio.load(str(audio_path))

            mapping: dict[str, str] = {}
            min_conf = cfg.speaker.remap_min_confidence

            for lbl, recs in label_records.items():
                embs = []
                for rec in recs:
                    dur = rec["end"] - rec["start"]
                    if dur < cfg.speaker.min_segment_duration_s:
                        continue
                    s     = int(rec["start"] * sr)
                    e     = int(rec["end"]   * sr)
                    chunk = waveform[:, s:e]
                    try:
                        emb = inference({"waveform": chunk.unsqueeze(0),
                                         "sample_rate": sr})
                        embs.append(np.array(emb))
                    except Exception:
                        pass

                if not embs:
                    continue

                centroid  = np.mean(embs, axis=0)
                name, conf = speaker_manager.match_speaker(centroid)
                if name and conf >= min_conf:
                    mapping[lbl] = name
                    print(f"[WhisperX]   {lbl} → {name} (confidence {conf:.3f})")

            # Apply mapping
            for rec in records:
                if rec["speaker"] in mapping:
                    rec["speaker"] = mapping[rec["speaker"]]

        except Exception as ex:
            print(f"[WhisperX] Remap failed: {ex}")

        return records

    def _write_pending_mapping(
        self,
        audio_path: Path,
        diarize_records: list[dict],
    ) -> None:
        """
        Write pending_<stem>.json only for labels that are still SPEAKER_XX
        (i.e. not yet matched to a known person).
        Sample segments are included so the user can listen and identify who's who.
        """
        samples: dict[str, list[str]] = {}
        for rec in diarize_records:
            lbl = rec["speaker"]
            # Only include unresolved auto-labels
            if not (lbl.startswith("SPEAKER_") or lbl == "SPEAKER_UNKNOWN"):
                continue
            samples.setdefault(lbl, [])
            if len(samples[lbl]) < 4:
                samples[lbl].append(
                    f"{fmt_srt(rec['start'])} --> {fmt_srt(rec['end'])}"
                )

        if not samples:
            # All speakers were matched — no pending mapping needed
            return

        mapping = {
            "source_file":  audio_path.name,
            "instructions": (
                "Listen to the sample_segments timestamps in the original audio "
                "to identify each speaker, then fill in 'real_name'. "
                "Run: python speaker_setup.py apply  (or use the Speaker Setup tab)"
            ),
            "speakers": {
                lbl: {"real_name": "", "sample_segments": segs}
                for lbl, segs in samples.items()
            },
        }

        path = cfg.paths.pending_mapping_dir / f"pending_{audio_path.stem}.json"
        path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\n[WhisperX] {len(samples)} speaker(s) need naming.")
        print(f"  → {path}")
        print(f"  → Fill in real_name, then: python speaker_setup.py apply")

    # ------------------------------------------------------------------
    # ASR & alignment
    # ------------------------------------------------------------------

    def _load_asr(self):
        if self._asr_model is None:
            import whisperx
            from diarization_postprocessor import DiarizationPostProcessor

            vad_overrides = DiarizationPostProcessor().get_vad_overrides() or {}
            vad_opts = {
                "chunk_size_s": self._cfg.chunk_size_s,
                "vad_onset":    self._cfg.vad_onset,
                "vad_offset":   self._cfg.vad_offset,
                **vad_overrides,
            }
            if vad_overrides:
                print(f"[WhisperX] Scene preset '{cfg.diarization_post.scene_preset}' "
                      f"overrides VAD: {vad_overrides}")

            print(f"[WhisperX] Loading ASR model {self._cfg.model_size} ...")
            self._asr_model = whisperx.load_model(
                self._cfg.model_size,
                self._dev.device,
                compute_type=self._cfg.compute_type,
                asr_options={"beam_size": self._cfg.beam_size},
                vad_options=vad_opts,
            )
            print("[WhisperX] ASR model loaded.")
        return self._asr_model

    def _load_align(self):
        if self._align_model is None:
            import whisperx
            print("[WhisperX] Loading alignment model ...")
            self._align_model, self._align_meta = whisperx.load_align_model(
                language_code=self._cfg.language,
                device=self._dev.device,
            )
            print("[WhisperX] Alignment model loaded.")
        return self._align_model, self._align_meta

    def _run_asr(self, audio_path: Path) -> dict:
        model = self._load_asr()
        print("[WhisperX] Running ASR ...")
        return model.transcribe(
            str(audio_path),
            batch_size=self._cfg.batch_size,
            language=self._cfg.language,
        )

    def _run_align(self, segments: list, audio_path: Path) -> dict:
        align_model, metadata = self._load_align()
        print("[WhisperX] Running alignment ...")
        import whisperx
        return whisperx.align(
            segments, align_model, metadata,
            str(audio_path), self._dev.device,
            return_char_alignments=self._cfg.return_char_alignments,
        )

    # ------------------------------------------------------------------
    # Word assignment & conversion
    # ------------------------------------------------------------------

    def _assign_words(self, aligned: dict, diarize_records: list[dict]) -> list[dict]:
        """Assign speaker labels to aligned word segments."""
        try:
            from pyannote.core import Annotation, Segment as PySeg
            annotation = Annotation()
            for rec in diarize_records:
                annotation[PySeg(rec["start"], rec["end"])] = rec["speaker"]

            import whisperx
            result = whisperx.assign_word_speakers(annotation, aligned)
            return result["segments"]
        except Exception as ex:
            print(f"[WhisperX] Word assignment error: {ex} — returning unassigned.")
            return aligned.get("segments", [])

    @staticmethod
    def _to_segments(raw_segments: list[dict]) -> list[Segment]:
        return [
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s.get("text", "").strip(),
                speaker=s.get("speaker", "SPEAKER_UNKNOWN"),
                source="whisperx",
                words=s.get("words", []),
            )
            for s in raw_segments
        ]
