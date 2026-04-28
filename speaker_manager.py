"""
speaker_manager.py
==================
Speaker voiceprint database — storage, weighted clustering, remapping, correction.

Storage layout per speaker  (DATA_ROOT/speakers/<Name>.json):
{
  "name": "Alice",
  "anchors": [                          # hand-labelled, never rolled out
    {"vector": [...], "source_file": "A.mp3", "timestamp": "..."}
  ],
  "embeddings": [                       # auto-learned, subject to rolling window
    {
      "vector": [...],
      "source_file": "B.mp3",
      "file_weight": 0.8,               # per-embedding quality weight (0.0–1.0)
      "timestamp": "...",
      "is_correction": false
    }
  ],
  "soft_corrections": [                 # correction samples too far from centroid
    {"vector": [...], "source_file": "...", "timestamp": "..."}
  ],
  "correction_count": 0,
  "created_at": "...",
  "updated_at": "..."
}

Key concepts:
  - anchors      : gold standard, user-verified, permanent, weight=1.0
  - embeddings   : auto-extracted, rolling window, each has file_weight
  - soft_corrections : correction samples that are far from centroid;
                       used as auxiliary vote, not averaged into centroid
  - centroid     : anchor_blend * anchor_mean + (1-anchor_blend) * weighted_emb_mean
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config import cfg
from utils import Segment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def _weighted_mean(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray | None:
    if not vectors:
        return None
    total = sum(weights)
    if total < 1e-9:
        return np.mean(vectors, axis=0)
    return np.sum([v * w for v, w in zip(vectors, weights)], axis=0) / total


# ---------------------------------------------------------------------------
# SpeakerManager
# ---------------------------------------------------------------------------

class SpeakerManager:
    """
    Manages the on-disk speaker database and provides:
      - embedding extraction & storage
      - weighted centroid computation
      - speaker remapping (temp label → real name)
      - correction workflow (single segment or batch from SRT diff)
      - utility: bulk file_weight update across all speakers
    """

    def __init__(self) -> None:
        self._dir  = cfg.paths.speakers_dir
        self._scfg = cfg.speaker
        self._db: dict[str, dict] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public: database queries
    # ------------------------------------------------------------------

    def list_speakers(self) -> list[str]:
        return list(self._db.keys())

    def has_speakers(self) -> bool:
        return bool(self._db)

    def centroid(self, name: str) -> np.ndarray | None:
        """
        Compute the reference centroid vector for a speaker.
        centroid = anchor_blend * anchor_mean + (1-anchor_blend) * weighted_emb_mean
        Falls back gracefully if either pool is empty.
        """
        data = self._db.get(name)
        if data is None:
            return None

        blend = self._scfg.anchor_blend

        # Anchor mean (uniform weight — all anchors are gold standard)
        anchor_vecs = [np.array(e["vector"]) for e in data.get("anchors", [])]
        anchor_mean = np.mean(anchor_vecs, axis=0) if anchor_vecs else None

        # Weighted embedding mean
        emb_entries = data.get("embeddings", [])
        emb_vecs    = [np.array(e["vector"])      for e in emb_entries]
        emb_weights = [float(e.get("file_weight", 1.0)) for e in emb_entries]
        emb_mean    = _weighted_mean(emb_vecs, emb_weights) if emb_vecs else None

        if anchor_mean is not None and emb_mean is not None:
            return blend * anchor_mean + (1.0 - blend) * emb_mean
        if anchor_mean is not None:
            return anchor_mean
        return emb_mean

    def all_centroids(self) -> dict[str, np.ndarray]:
        """Return {name: centroid} for all speakers that have data."""
        result = {}
        for name in self._db:
            c = self.centroid(name)
            if c is not None:
                result[name] = c
        return result

    def match_speaker(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """
        Find the best matching known speaker for a given embedding.
        Returns (name, confidence) — name is None if below remap_min_confidence.
        Soft corrections act as auxiliary vote: if they agree with the best match,
        confidence gets a small boost; they cannot override the centroid decision.
        """
        best_name: str | None = None
        best_score: float = -1.0

        for name, centroid in self.all_centroids().items():
            score = _cos(embedding, centroid)
            if score > best_score:
                best_score = score
                best_name  = name

        # Auxiliary vote from soft_corrections
        if best_name is not None:
            soft = self._db[best_name].get("soft_corrections", [])
            if soft:
                soft_scores = [_cos(embedding, np.array(e["vector"])) for e in soft]
                avg_soft    = float(np.mean(soft_scores))
                # Small boost (max +0.05) if soft corrections agree
                if avg_soft > self._scfg.similarity_threshold:
                    best_score = min(1.0, best_score + 0.05 * (avg_soft - self._scfg.similarity_threshold))

        min_conf = self._scfg.remap_min_confidence
        if best_score >= min_conf:
            return best_name, best_score
        return None, best_score

    # ------------------------------------------------------------------
    # Public: embedding extraction
    # ------------------------------------------------------------------

    def extract_embeddings_for_segments(
        self,
        audio_path: Path | str,
        segments: list[dict],   # [{speaker, start, end}, ...]
    ) -> dict[str, list[np.ndarray]]:
        """
        Extract embeddings for a list of diarization segments.
        Returns {speaker_label: [embedding, ...]} using pyannote embedding model.
        """
        audio_path = Path(audio_path)
        result: dict[str, list[np.ndarray]] = {}

        try:
            from pyannote.audio import Model, Inference
            import torchaudio

            model     = Model.from_pretrained("pyannote/embedding",
                                              use_auth_token=cfg.hf_token)
            inference = Inference(model, window="whole")
            waveform, sr = torchaudio.load(str(audio_path))

            for seg in segments:
                speaker = seg["speaker"]
                start_s = float(seg["start"])
                end_s   = float(seg["end"])

                if end_s - start_s < self._scfg.min_segment_duration_s:
                    continue

                s     = int(start_s * sr)
                e     = int(end_s   * sr)
                chunk = waveform[:, s:e]
                emb   = inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr})
                result.setdefault(speaker, []).append(np.array(emb))

        except Exception as ex:
            print(f"[SpeakerManager] Embedding extraction failed: {ex}")

        return result

    # ------------------------------------------------------------------
    # Public: add anchors (gold standard, permanent)
    # ------------------------------------------------------------------

    def add_anchor(
        self,
        name: str,
        embedding: np.ndarray,
        source_file: str = "",
    ) -> None:
        """Add a gold-standard anchor embedding (never rolled out)."""
        self._ensure_speaker(name)
        self._db[name]["anchors"].append({
            "vector":      embedding.tolist(),
            "source_file": source_file,
            "timestamp":   _now(),
        })
        self._save(name)
        print(f"[SpeakerManager] Anchor added for '{name}' (total anchors: "
              f"{len(self._db[name]['anchors'])})")

    # ------------------------------------------------------------------
    # Public: store auto-learned embeddings after diarization
    # ------------------------------------------------------------------

    def store_embeddings(
        self,
        name: str,
        embeddings: list[np.ndarray],
        source_file: str = "",
        file_weight: float = 1.0,
    ) -> None:
        """
        Store auto-learned embeddings for a speaker.
        Applies rolling window if over max_embeddings_per_speaker.
        """
        self._ensure_speaker(name)
        for emb in embeddings:
            self._db[name]["embeddings"].append({
                "vector":       emb.tolist(),
                "source_file":  source_file,
                "file_weight":  file_weight,
                "timestamp":    _now(),
                "is_correction": False,
            })

        self._apply_rolling(name)
        self._save(name)

    # ------------------------------------------------------------------
    # Public: remapping (cold-start → real names)
    # ------------------------------------------------------------------

    def remap_labels(
        self,
        segments: list[dict],   # [{speaker, start, end}, ...]
        audio_path: Path | str,
    ) -> dict[str, str]:
        """
        Given diarized segments with temp labels (SPEAKER_00 etc.),
        compute per-label centroids and match against known speakers.
        Returns {temp_label: real_name_or_SPEAKER_NEW_XX}.

        Side effect: stores matched embeddings into the database.
        """
        audio_path = Path(audio_path)

        # Group embeddings by temp label
        per_label_embs = self.extract_embeddings_for_segments(audio_path, segments)
        if not per_label_embs:
            print("[SpeakerManager] No embeddings extracted — cannot remap.")
            return {}

        mapping: dict[str, str] = {}
        new_counter = 0

        for label, embs in per_label_embs.items():
            if not embs:
                continue
            centroid = np.mean(embs, axis=0)
            name, confidence = self.match_speaker(centroid)

            if name is not None:
                mapping[label] = name
                print(f"[SpeakerManager] {label} → {name}  (confidence: {confidence:.3f})")
                # Store into auto embeddings
                self.store_embeddings(
                    name, embs,
                    source_file=audio_path.name,
                    file_weight=1.0,  # default; user can adjust later
                )
            else:
                new_label = f"SPEAKER_NEW_{new_counter:02d}"
                mapping[label] = new_label
                new_counter += 1
                print(f"[SpeakerManager] {label} → {new_label}  "
                      f"(confidence {confidence:.3f} below threshold, needs naming)")

        return mapping

    # ------------------------------------------------------------------
    # Public: correction workflow
    # ------------------------------------------------------------------

    def correct_segment(
        self,
        audio_path: Path | str,
        segment: Segment,
        correct_speaker: str,
        confident: bool = False,
    ) -> None:
        """
        Correct a single misidentified segment.

        confident=True  → sample is clean; add directly to embeddings
        confident=False → compute distance from centroid first:
                          - close  → add to embeddings (with lower weight)
                          - far    → add to soft_corrections only
        """
        audio_path = Path(audio_path)
        emb = self._extract_single(audio_path, segment.start, segment.end)
        if emb is None:
            print(f"[SpeakerManager] Could not extract embedding for correction.")
            return

        self._ensure_speaker(correct_speaker)
        wrong = segment.speaker

        if confident:
            # User is sure — add directly to embeddings
            self.store_embeddings(
                correct_speaker, [emb],
                source_file=audio_path.name,
                file_weight=0.8,   # slightly below 1.0 — still correction
            )
            print(f"[SpeakerManager] Confident correction: "
                  f"{wrong} → {correct_speaker} (added to embeddings)")
        else:
            # Measure distance from current centroid
            c = self.centroid(correct_speaker)
            dist = 1.0 - _cos(emb, c) if c is not None else 1.0
            thresh = self._scfg.correction_distance_threshold

            if dist <= thresh:
                # Close to centroid — trustworthy, add to embeddings
                self.store_embeddings(
                    correct_speaker, [emb],
                    source_file=audio_path.name,
                    file_weight=0.6,
                )
                print(f"[SpeakerManager] Correction (close, dist={dist:.3f}): "
                      f"{wrong} → {correct_speaker} (added to embeddings)")
            else:
                # Far from centroid — possibly noisy; keep as soft correction
                self._db[correct_speaker].setdefault("soft_corrections", []).append({
                    "vector":      emb.tolist(),
                    "source_file": audio_path.name,
                    "timestamp":   _now(),
                })
                print(f"[SpeakerManager] Correction (far, dist={dist:.3f}): "
                      f"{wrong} → {correct_speaker} (added to soft_corrections)")

        # Remove closest embedding from wrong speaker if they had one
        if wrong in self._db and wrong != correct_speaker:
            self._remove_closest_embedding(wrong, emb)
            self._db[wrong]["correction_count"] = (
                self._db[wrong].get("correction_count", 0) + 1
            )

        self._db[correct_speaker]["correction_count"] = (
            self._db[correct_speaker].get("correction_count", 0) + 1
        )
        self._save(correct_speaker)
        if wrong in self._db and wrong != correct_speaker:
            self._save(wrong)

    def apply_corrections_from_srt_diff(
        self,
        audio_path: Path | str,
        corrected_segs: list[Segment],
        original_segs: list[Segment],
        confident: bool = False,
    ) -> None:
        """Batch correction: compare original vs corrected segments, fix changed speakers."""
        changed = 0
        for orig, corr in zip(original_segs, corrected_segs):
            if orig.speaker != corr.speaker and corr.speaker:
                self.correct_segment(audio_path, orig, corr.speaker, confident=confident)
                changed += 1
        print(f"[SpeakerManager] Batch correction done: {changed} segments updated.")

    # ------------------------------------------------------------------
    # Public: utility — bulk file_weight update
    # ------------------------------------------------------------------

    def set_file_weight(self, source_file: str, weight: float) -> None:
        """
        Update file_weight for all embeddings from a given source file,
        across all speakers. Useful when you decide a recording was low quality.

        weight=0.0 effectively excludes all embeddings from that file
        from the centroid calculation without deleting them.
        """
        updated = 0
        for name, data in self._db.items():
            for entry in data.get("embeddings", []):
                if entry.get("source_file") == source_file:
                    entry["file_weight"] = weight
                    updated += 1
            self._save(name)
        print(f"[SpeakerManager] set_file_weight('{source_file}', {weight}): "
              f"{updated} entries updated across all speakers.")

    def drop_file_embeddings(self, source_file: str) -> None:
        """
        Permanently remove all embeddings from a given source file
        across all speakers. Use when you're sure the file was unusable.
        """
        removed = 0
        for name, data in self._db.items():
            before = len(data.get("embeddings", []))
            data["embeddings"] = [
                e for e in data.get("embeddings", [])
                if e.get("source_file") != source_file
            ]
            removed += before - len(data["embeddings"])
            self._save(name)
        print(f"[SpeakerManager] drop_file_embeddings('{source_file}'): "
              f"{removed} entries removed.")

    def print_db_summary(self) -> None:
        """Print a summary of the speaker database."""
        if not self._db:
            print("[SpeakerManager] Database is empty.")
            return
        print(f"\n{'='*55}")
        print(f"  Speaker Database ({len(self._db)} speakers)")
        print(f"{'='*55}")
        for name, data in self._db.items():
            anchors     = len(data.get("anchors", []))
            embeddings  = len(data.get("embeddings", []))
            soft_corr   = len(data.get("soft_corrections", []))
            corrections = data.get("correction_count", 0)
            c = self.centroid(name)
            print(f"  {name:<20} anchors={anchors:3d}  emb={embeddings:3d}  "
                  f"soft={soft_corr:3d}  corrections={corrections:3d}  "
                  f"centroid={'ok' if c is not None else 'NONE'}")
        print(f"{'='*55}\n")

    # ------------------------------------------------------------------
    # Private: I/O
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        self._db = {}
        for fp in self._dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._db[data["name"]] = data
            except Exception as ex:
                print(f"[SpeakerManager] Failed to load {fp.name}: {ex}")
        if self._db:
            print(f"[SpeakerManager] Loaded {len(self._db)} speaker(s): "
                  f"{list(self._db.keys())}")

    def _save(self, name: str) -> None:
        data = self._db.get(name)
        if data is None:
            return
        data["updated_at"] = _now()
        safe = name.replace(" ", "_").replace("/", "_")
        fp   = self._dir / f"{safe}.json"
        fp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_all(self) -> None:
        for name in self._db:
            self._save(name)

    def _ensure_speaker(self, name: str) -> None:
        if name not in self._db:
            self._db[name] = {
                "name":             name,
                "anchors":          [],
                "embeddings":       [],
                "soft_corrections": [],
                "correction_count": 0,
                "created_at":       _now(),
                "updated_at":       _now(),
            }

    def _apply_rolling(self, name: str) -> None:
        """Enforce max_embeddings_per_speaker with rolling or keep_first strategy."""
        max_n    = self._scfg.max_embeddings_per_speaker
        strategy = self._scfg.embedding_strategy
        existing = self._db[name]["embeddings"]
        if len(existing) > max_n:
            self._db[name]["embeddings"] = (
                existing[:max_n] if strategy == "keep_first" else existing[-max_n:]
            )

    def _remove_closest_embedding(self, name: str, target: np.ndarray) -> None:
        embs = self._db[name].get("embeddings", [])
        if not embs:
            return
        scores = [_cos(np.array(e["vector"]), target) for e in embs]
        embs.pop(int(np.argmax(scores)))

    # ------------------------------------------------------------------
    # Private: single embedding extraction
    # ------------------------------------------------------------------

    def _extract_single(
        self, audio_path: Path, start_s: float, end_s: float
    ) -> np.ndarray | None:
        try:
            from pyannote.audio import Model, Inference
            import torchaudio

            model     = Model.from_pretrained("pyannote/embedding",
                                              use_auth_token=cfg.hf_token)
            inference = Inference(model, window="whole")
            waveform, sr = torchaudio.load(str(audio_path))
            chunk = waveform[:, int(start_s * sr): int(end_s * sr)]
            return np.array(inference({"waveform": chunk.unsqueeze(0),
                                       "sample_rate": sr}))
        except Exception as ex:
            print(f"[SpeakerManager] Single extraction failed: {ex}")
            return None
