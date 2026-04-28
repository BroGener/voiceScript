"""
transcriber_whisperx.py
=======================
WhisperX transcriber with Route B diarization:
  ASR → time alignment → segmentation (pyannote) → custom clustering
  → speaker remapping using historical voiceprint database

Route B vs original DiarizationPipeline:
  Original: audio → black-box pipeline → SPEAKER_00/01 (no external anchors)
  Route B:  audio → segmentation only → extract embeddings per segment
            → cluster new embeddings TOGETHER with historical centroids
            → assign real speaker names directly

Cold-start (empty database):
  Falls back to DiarizationPipeline → writes pending_mapping.json
  → user runs speaker_setup.py to name speakers and seed the database
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
from utils import Segment


class WhisperXTranscriber:

    def __init__(self) -> None:
        self._asr_model     = None
        self._align_model   = None
        self._align_meta    = None
        self._seg_model     = None      # segmentation model (Route B)
        self._embed_model   = None      # embedding model (Route B)
        self._diarize_model = None      # fallback DiarizationPipeline (cold-start)
        self._cfg = cfg.whisperx
        self._dev = cfg.device

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path | str,
        speaker_manager=None,
    ) -> list[Segment]:
        """
        Full pipeline: ASR → alignment → diarization → speaker assignment.

        speaker_manager: SpeakerManager instance.
          - If it has known speakers → Route B (constrained clustering)
          - If empty                 → cold-start fallback
        """
        audio_path = Path(audio_path)
        print(f"\n[WhisperX] Transcribing: {audio_path.name}")

        raw     = self._run_asr(audio_path)
        aligned = self._run_align(raw["segments"], audio_path)

        has_speakers = (speaker_manager is not None and
                        speaker_manager.has_speakers())

        if has_speakers:
            segments = self._route_b_diarize(audio_path, aligned, speaker_manager)
        else:
            segments = self._coldstart_diarize(audio_path, aligned, speaker_manager)

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
    # Route B: constrained clustering using historical voiceprints
    # ------------------------------------------------------------------

    def _route_b_diarize(
        self,
        audio_path: Path,
        aligned: dict,
        speaker_manager,
    ) -> list[Segment]:
        """
        Diarize using pyannote segmentation + our own clustering.
        Historical speaker centroids act as fixed anchor points in the
        embedding space, pulling new embeddings toward the correct identity.
        """
        print("[WhisperX] Route B: segmentation + constrained clustering ...")

        # Step 1: Run pyannote segmentation (boundary detection only, no clustering)
        raw_segments = self._run_segmentation(audio_path)
        if not raw_segments:
            print("[WhisperX] Segmentation returned no segments — falling back.")
            return self._coldstart_diarize(audio_path, aligned, speaker_manager)

        # Step 2: Extract embedding for each segment
        embeddings, valid_segs = self._extract_segment_embeddings(audio_path, raw_segments)
        if not embeddings:
            print("[WhisperX] No embeddings extracted — falling back.")
            return self._coldstart_diarize(audio_path, aligned, speaker_manager)

        # Step 3: Constrained clustering
        #   Build combined matrix: known centroids (fixed) + new embeddings
        #   Run hierarchical clustering, keep centroid assignments fixed
        speaker_labels = self._constrained_cluster(embeddings, speaker_manager)

        # Step 4: Apply post-processing (merge gaps, remove short segments)
        diarize_records = [
            {"speaker": lbl, "start": seg["start"], "end": seg["end"]}
            for lbl, seg in zip(speaker_labels, valid_segs)
        ]
        post = DiarizationPostProcessor()
        diarize_records = post.process_records(diarize_records)

        # Step 5: Assign words to speakers
        segments = self._assign_words(aligned, diarize_records)
        return self._to_segments(segments)

    def _run_segmentation(self, audio_path: Path) -> list[dict]:
        """
        Run pyannote segmentation model to get speech boundaries.
        Returns [{start, end}, ...] without speaker labels.
        """
        if self._seg_model is None:
            from pyannote.audio import Model, Inference
            print("[WhisperX] Loading segmentation model ...")
            self._seg_model = Model.from_pretrained(
                "pyannote/segmentation",
                use_auth_token=cfg.hf_token,
            )
            print("[WhisperX] Segmentation model loaded.")

        from pyannote.audio import Inference
        inference = Inference(self._seg_model, window="sliding",
                              duration=5.0, step=0.5)

        try:
            output = inference(str(audio_path))
            # output is a SlidingWindowFeature; convert to segment list
            segments = []
            # Each frame above threshold is speech; merge contiguous blocks
            scores   = output.data.max(axis=1)  # max across speakers dim
            frames   = output.sliding_window
            threshold = 0.5

            in_speech  = False
            seg_start  = 0.0

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
                segments.append({"start": seg_start,
                                  "end": frames[-1].end})

            print(f"[WhisperX] Segmentation: {len(segments)} speech segments.")
            return segments

        except Exception as ex:
            print(f"[WhisperX] Segmentation error: {ex}")
            return []

    def _extract_segment_embeddings(
        self,
        audio_path: Path,
        segments: list[dict],
    ) -> tuple[list[np.ndarray], list[dict]]:
        """
        Extract one embedding per segment.
        Returns (embeddings, valid_segments) — segments too short are skipped.
        """
        if self._embed_model is None:
            from pyannote.audio import Model, Inference
            print("[WhisperX] Loading embedding model ...")
            self._embed_model = Model.from_pretrained(
                "pyannote/embedding",
                use_auth_token=cfg.hf_token,
            )
            print("[WhisperX] Embedding model loaded.")

        from pyannote.audio import Inference
        import torchaudio

        inference = Inference(self._embed_model, window="whole")
        waveform, sr = torchaudio.load(str(audio_path))

        embeddings: list[np.ndarray] = []
        valid_segs: list[dict]       = []
        min_dur = cfg.speaker.min_segment_duration_s

        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration < min_dur:
                continue
            s     = int(seg["start"] * sr)
            e     = int(seg["end"]   * sr)
            chunk = waveform[:, s:e]
            try:
                emb = inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr})
                embeddings.append(np.array(emb))
                valid_segs.append(seg)
            except Exception as ex:
                print(f"[WhisperX] Embedding failed for segment "
                      f"{seg['start']:.1f}-{seg['end']:.1f}: {ex}")

        print(f"[WhisperX] Extracted {len(embeddings)} embeddings "
              f"({len(segments) - len(embeddings)} skipped).")
        return embeddings, valid_segs

    def _constrained_cluster(
        self,
        new_embeddings: list[np.ndarray],
        speaker_manager,
    ) -> list[str]:
        """
        Cluster new embeddings with known centroids as fixed anchor points.

        Strategy:
          1. Compute cosine distance from each new embedding to every known centroid.
          2. If best match >= remap_min_confidence → assign that speaker directly.
          3. Remaining unmatched embeddings → hierarchical clustering among themselves
             → label as SPEAKER_NEW_XX.

        This avoids the 'sample imbalance' problem: known centroids are stable
        regardless of how many new samples each speaker has in this recording.
        """
        centroids = speaker_manager.all_centroids()
        min_conf  = cfg.speaker.remap_min_confidence
        labels: list[str] = [""] * len(new_embeddings)
        unmatched_indices: list[int] = []

        # Pass 1: match against known centroids
        for i, emb in enumerate(new_embeddings):
            best_name, best_score = speaker_manager.match_speaker(emb)
            if best_name is not None and best_score >= min_conf:
                labels[i] = best_name
            else:
                unmatched_indices.append(i)

        # Pass 2: cluster unmatched embeddings among themselves
        if unmatched_indices:
            unmatched_vecs = np.stack([new_embeddings[i] for i in unmatched_indices])
            if len(unmatched_vecs) == 1:
                cluster_labels = [0]
            else:
                dist_matrix  = pdist(unmatched_vecs, metric="cosine")
                Z            = linkage(dist_matrix, method="average")
                n_new        = max(1, self._cfg.max_speakers - len(centroids))
                cluster_labels = fcluster(Z, t=n_new, criterion="maxclust").tolist()

            for idx, cluster_id in zip(unmatched_indices, cluster_labels):
                labels[idx] = f"SPEAKER_NEW_{cluster_id - 1:02d}"

        matched   = sum(1 for l in labels if not l.startswith("SPEAKER_NEW"))
        unmatched = len(labels) - matched
        print(f"[WhisperX] Clustering: {matched} matched to known speakers, "
              f"{unmatched} new/unknown segments.")
        return labels

    # ------------------------------------------------------------------
    # Cold-start: no database yet → use DiarizationPipeline
    # ------------------------------------------------------------------

    def _coldstart_diarize(
        self,
        audio_path: Path,
        aligned: dict,
        speaker_manager,
    ) -> list[Segment]:
        """
        Cold-start: run full DiarizationPipeline (no external anchors).
        Writes pending_mapping.json for the user to fill in speaker names.
        After the user runs speaker_setup.py, the database is seeded and
        subsequent runs use Route B.
        """
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

        raw_annotation = self._diarize_model(str(audio_path), **kwargs)

        post = DiarizationPostProcessor()
        annotation = post.process(raw_annotation)

        # Convert annotation to records
        diarize_records = []
        try:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                diarize_records.append({
                    "speaker": speaker,
                    "start":   float(turn.start),
                    "end":     float(turn.end),
                })
        except AttributeError:
            pass

        # Write pending_mapping.json so user can name the speakers
        self._write_pending_mapping(audio_path, diarize_records)

        segments = self._assign_words(aligned, diarize_records)
        return self._to_segments(segments)

    def _write_pending_mapping(
        self,
        audio_path: Path,
        diarize_records: list[dict],
    ) -> None:
        """
        Write pending_mapping.json with temp labels and sample timestamps.
        User fills in real names, then runs speaker_setup.py apply.
        """
        # Collect sample segments per label
        samples: dict[str, list[str]] = {}
        for rec in diarize_records:
            lbl = rec["speaker"]
            if lbl not in samples:
                samples[lbl] = []
            if len(samples[lbl]) < 3:
                from utils import fmt_srt
                samples[lbl].append(
                    f"{fmt_srt(rec['start'])} --> {fmt_srt(rec['end'])}"
                )

        mapping = {
            "source_file":    audio_path.name,
            "instructions":   (
                "Fill in 'real_name' for each speaker label, "
                "then run: python speaker_setup.py apply"
            ),
            "speakers": {
                lbl: {"real_name": "", "sample_segments": segs}
                for lbl, segs in samples.items()
            },
        }

        path = cfg.paths.pending_mapping_file
        path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        print(f"\n[WhisperX] Cold-start complete.")
        print(f"  → SRT written with temp labels (SPEAKER_00, SPEAKER_01 ...)")
        print(f"  → Open {path} and fill in speaker names")
        print(f"  → Then run: python speaker_setup.py apply")

    # ------------------------------------------------------------------
    # ASR & alignment (unchanged from before)
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
    # Word assignment & format conversion
    # ------------------------------------------------------------------

    def _assign_words(self, aligned: dict, diarize_records: list[dict]) -> list[dict]:
        """
        Assign speaker labels from diarize_records to aligned word segments.
        Mimics whisperx.assign_word_speakers but works with our record format.
        """
        try:
            # Convert our records to pyannote Annotation for assign_word_speakers
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
