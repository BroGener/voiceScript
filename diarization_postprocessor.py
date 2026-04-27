"""
diarization_postprocessor.py
=============================
说话人分割后处理器。

解决 pyannote 在"对话密集/快速交替"场景下的典型问题：

  问题根源：pyannote 用 VAD onset/offset 双阈值判断发言边界。
  两人快速交替时，边界检测不准，导致相邻短片段被合并归给同一人。
  这发生在声纹匹配之前，所以声纹数据库再准也无法修复。

  本模块在 pyannote 原始输出之后、assign_word_speakers 之前，
  插入纯规则后处理，不需要修改 whisperx 内部。

四步处理流程：
  1. 重叠解决   — 两人片段时间重叠时，按策略裁剪边界
  2. 短片段移除 — 极短孤立片段（< min_segment_s）并入最近邻居
  3. 相邻合并   — 同一说话人间隔 < merge_gap_s 的片段合并（核心）
  4. 再次清理   — 合并后可能产生新碎片，再扫一遍

场景预设（VAD 参数覆盖）：
  "dialogue" → onset/offset 降到 0.15，chunk_size 缩到 15s
              更敏感地感知换人边界，产生更多初始碎片，
              但合并步骤会把碎片重新整理好
  "lecture"  → onset/offset 升到 0.5，减少呼吸/停顿引起的误切
"""

from __future__ import annotations
from dataclasses import dataclass

from config import cfg


@dataclass
class _Seg:
    """内部轻量分割段。"""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


class DiarizationPostProcessor:

    def __init__(self) -> None:
        self._cfg = cfg.diarization_post

    # ── 公开接口 ─────────────────────────────────────────────

    def process(self, annotation):
        """
        接收 pyannote Annotation，返回后处理过的 Annotation。
        cfg.diarization_post.enabled=False 时原样返回。
        """
        if not self._cfg.enabled:
            return annotation

        segs = self._from_annotation(annotation)
        if not segs:
            return annotation

        before = len(segs)

        segs = self._resolve_overlaps(segs)
        segs = self._remove_short(segs)
        segs = self._merge_adjacent(segs)
        segs = self._remove_short(segs)   # 合并后再清一遍

        print(f"[DiarizPost] {before} 片段 → {len(segs)} 片段 "
              f"(清理了 {before - len(segs)} 个碎片/重叠)")

        return self._to_annotation(segs, annotation)

    def get_vad_overrides(self) -> dict | None:
        """
        根据场景预设返回需要覆盖的 VAD 参数。
        返回 None 表示使用默认配置。
        """
        preset = self._cfg.scene_preset
        if preset == "dialogue":
            return {
                "vad_onset":   0.15,
                "vad_offset":  0.15,
                "chunk_size_s": 15.0,
            }
        if preset == "lecture":
            return {
                "vad_onset":   0.5,
                "vad_offset":  0.5,
                "chunk_size_s": 30.0,
            }
        return None

    # ── 私有：四步处理 ───────────────────────────────────────

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

            # 有重叠
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
        """移除极短孤立片段，并入最近邻居而非直接删除。"""
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
                        result[-1] = _Seg(result[-1].start, max(result[-1].end, seg.end), result[-1].speaker)
                    elif next_s:
                        segs[i + 1] = _Seg(min(seg.start, next_s.start), next_s.end, next_s.speaker)
                    changed = True
                else:
                    result.append(seg)
                i += 1
            segs = result

        return segs

    def _merge_adjacent(self, segs: list[_Seg]) -> list[_Seg]:
        """合并同一说话人的相邻片段（间隔 < merge_gap_s）。"""
        gap = self._cfg.merge_gap_s
        segs = sorted(segs, key=lambda s: s.start)
        result: list[_Seg] = []

        for seg in segs:
            if result and result[-1].speaker == seg.speaker and (seg.start - result[-1].end) <= gap:
                result[-1] = _Seg(result[-1].start, seg.end, result[-1].speaker)
            else:
                result.append(seg)

        return result

    # ── 私有：pyannote 格式转换 ──────────────────────────────

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
