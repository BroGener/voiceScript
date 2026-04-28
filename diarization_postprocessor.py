"""
diarization_postprocessor.py
=============================
Post-processing for pyannote diarization output.
Works with both pyannote Annotation objects (cold-start)
and plain record lists (Route B).

Four-step pipeline:
  1. Resolve overlaps
  2. Remove short isolated segments
  3. Merge adjacent same-speaker segments
  4. Second pass short segment removal
"""

from __future__ import annotations
from dataclasses import dataclass

from config import cfg


@dataclass
class _Seg:
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiarizationPostProcessor:

    def __init__(self) -> None:
        self._cfg = cfg.diarization_post

    # ------------------------------------------------------------------
    # Public: process pyannote Annotation (cold-start path)
    # ------------------------------------------------------------------

    def process(self, annotation):
        """Process a pyannote Annotation object."""
        if not self._cfg.enabled:
            return annotation

        segs = self._from_annotation(annotation)
        if not segs:
            return annotation

        segs = self._run_pipeline(segs)
        return self._to_annotation(segs, annotation)

    # ------------------------------------------------------------------
    # Public: process plain record list (Route B path)
    # ------------------------------------------------------------------

    def process_records(self, records: list[dict]) -> list[dict]:
        """
        Process a list of {speaker, start, end} dicts.
        Returns the same format after post-processing.
        """
        if not self._cfg.enabled or not records:
            return records

        segs = [_Seg(float(r["start"]), float(r["end"]), r["speaker"])
                for r in records]
        segs = self._run_pipeline(segs)
        return [{"speaker": s.speaker, "start": s.start, "end": s.end}
                for s in segs]

    # ------------------------------------------------------------------
    # VAD override for scene presets
    # ------------------------------------------------------------------

    def get_vad_overrides(self) -> dict | None:
        preset = self._cfg.scene_preset
        if preset == "dialogue":
            return {"vad_onset": 0.15, "vad_offset": 0.15, "chunk_size_s": 15.0}
        if preset == "lecture":
            return {"vad_onset": 0.5,  "vad_offset": 0.5,  "chunk_size_s": 30.0}
        return None

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, segs: list[_Seg]) -> list[_Seg]:
        before = len(segs)
        segs = self._resolve_overlaps(segs)
        segs = self._remove_short(segs)
        segs = self._merge_adjacent(segs)
        segs = self._remove_short(segs)
        print(f"[DiarizPost] {before} → {len(segs)} segments "
              f"({before - len(segs)} removed/merged)")
        return segs

    def _resolve_overlaps(self, segs: list[_Seg]) -> list[_Seg]:
        strategy = self._cfg.overlap_strategy
        if strategy == "keep":
            return segs

        segs = sorted(segs, key=lambda s: s.start)
        result: list[_Seg] = []

        for seg in segs:
            if not result:
                result.append(seg)
                continue

            prev = result[-1]
            if seg.start >= prev.end:
                result.append(seg)
                continue

            overlap_end = min(seg.end, prev.end)
            if strategy == "longer":
                if seg.duration >= prev.duration:
                    result[-1] = _Seg(prev.start, seg.start, prev.speaker)
                else:
                    seg = _Seg(overlap_end, seg.end, seg.speaker)
            elif strategy == "earlier":
                seg = _Seg(overlap_end, seg.end, seg.speaker)

            if seg.end > seg.start:
                result.append(seg)

        return result

    def _remove_short(self, segs: list[_Seg]) -> list[_Seg]:
        min_dur = self._cfg.min_segment_s
        segs = sorted(segs, key=lambda s: s.start)
        changed = True

        while changed:
            changed = False
            result: list[_Seg] = []
            i = 0
            while i < len(segs):
                seg = segs[i]
                if seg.duration < min_dur:
                    prev_s = result[-1] if result else None
                    next_s = segs[i + 1] if i + 1 < len(segs) else None

                    if prev_s is None and next_s is None:
                        i += 1
                        continue

                    use_prev = False
                    if prev_s and next_s:
                        use_prev = (seg.start - prev_s.end) <= (next_s.start - seg.end)
                    elif prev_s:
                        use_prev = True

                    if use_prev and result:
                        result[-1] = _Seg(result[-1].start,
                                          max(result[-1].end, seg.end),
                                          result[-1].speaker)
                    elif next_s:
                        segs[i + 1] = _Seg(min(seg.start, next_s.start),
                                           next_s.end,
                                           next_s.speaker)
                    changed = True
                else:
                    result.append(seg)
                i += 1
            segs = result

        return segs

    def _merge_adjacent(self, segs: list[_Seg]) -> list[_Seg]:
        gap    = self._cfg.merge_gap_s
        segs   = sorted(segs, key=lambda s: s.start)
        result: list[_Seg] = []

        for seg in segs:
            if (result and result[-1].speaker == seg.speaker
                    and (seg.start - result[-1].end) <= gap):
                result[-1] = _Seg(result[-1].start, seg.end, result[-1].speaker)
            else:
                result.append(seg)

        return result

    # ------------------------------------------------------------------
    # Format conversion (pyannote Annotation)
    # ------------------------------------------------------------------

    @staticmethod
    def _from_annotation(annotation) -> list[_Seg]:
        segs = []
        try:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                segs.append(_Seg(float(turn.start), float(turn.end), str(speaker)))
        except AttributeError:
            pass
        return sorted(segs, key=lambda s: s.start)

    @staticmethod
    def _to_annotation(segs: list[_Seg], original):
        try:
            from pyannote.core import Annotation, Segment as PySeg
            result = Annotation(uri=original.uri)
            for s in segs:
                if s.end > s.start:
                    result[PySeg(s.start, s.end)] = s.speaker
            return result
        except ImportError:
            return original
