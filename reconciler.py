"""
reconciler.py
=============
双模型结果校对合并器。

策略：
  - 时间轴以 Whisper 为基准（更贴近真实时间）
  - 在容差范围内找到对应的 WhisperX 片段
  - 说话人标签始终来自 WhisperX（Whisper 无此信息）
  - 文本一致 → 采用配置中的首选模型
  - 文本冲突 → 根据 conflict_strategy 保留单方或双方
  - 无法匹配的孤立片段 → 保留原文，标注来源
"""

from __future__ import annotations
from config import cfg
from utils import Segment


class Reconciler:

    def __init__(self) -> None:
        self._cfg = cfg.reconciler

    def reconcile(
        self,
        whisper_segs: list[Segment],
        whisperx_segs: list[Segment],
    ) -> list[Segment]:
        print(f"\n[Reconciler] 校对: Whisper={len(whisper_segs)} 片段, "
              f"WhisperX={len(whisperx_segs)} 片段")

        merged: list[Segment] = []
        used_wx: set[int] = set()

        for w_seg in whisper_segs:
            matches = self._find_matches(w_seg, whisperx_segs, used_wx)
            if not matches:
                merged.append(w_seg)
                continue
            best_idx, wx_seg = matches[0]
            used_wx.add(best_idx)
            merged.append(self._merge_pair(w_seg, wx_seg))

        # WhisperX 独有的片段（含说话人信息，有价值）
        for i, wx_seg in enumerate(whisperx_segs):
            if i not in used_wx:
                merged.append(wx_seg)

        merged.sort(key=lambda s: s.start)

        conflicts   = sum(1 for s in merged if s.source == "both")
        reconciled  = sum(1 for s in merged if s.source == "reconciled")
        w_only      = sum(1 for s in merged if s.source == "whisper")
        wx_only     = sum(1 for s in merged if s.source == "whisperx")
        print(f"[Reconciler] 完成: 总计 {len(merged)} 片段 | "
              f"合并 {reconciled} | 冲突 {conflicts} | "
              f"Whisper独有 {w_only} | WhisperX独有 {wx_only}")
        return merged

    def _find_matches(
        self,
        target: Segment,
        candidates: list[Segment],
        exclude: set[int],
    ) -> list[tuple[int, Segment]]:
        tol = self._cfg.time_tolerance_s
        matches = []
        for i, cand in enumerate(candidates):
            if i in exclude:
                continue
            overlap = min(target.end, cand.end) - max(target.start, cand.start)
            center_dist = abs(
                (target.start + target.end) / 2 - (cand.start + cand.end) / 2
            )
            if overlap > 0 or center_dist <= tol:
                matches.append((i, cand, max(overlap, 0)))
        matches.sort(key=lambda x: x[2], reverse=True)
        return [(idx, seg) for idx, seg, _ in matches]

    def _merge_pair(self, w: Segment, wx: Segment) -> Segment:
        speaker = wx.speaker if wx.speaker != "SPEAKER_UNKNOWN" else w.speaker
        text_w  = w.text.strip()
        text_wx = wx.text.strip()

        if text_w.lower() == text_wx.lower():
            text = text_wx if self._cfg.conflict_strategy != "prefer_whisper" else text_w
            return Segment(
                start=w.start, end=w.end,
                text=text, speaker=speaker,
                source="reconciled",
                words=wx.words or w.words,
                extra={**w.extra, **wx.extra},
            )

        strategy = self._cfg.conflict_strategy
        if strategy == "prefer_whisper":
            return Segment(start=w.start, end=w.end, text=w.text,
                           speaker=speaker, source="whisper",
                           words=w.words, extra=w.extra)
        if strategy == "prefer_whisperx":
            return Segment(start=w.start, end=w.end, text=wx.text,
                           speaker=speaker, source="whisperx",
                           words=wx.words, extra=wx.extra)
        # keep_both（默认）
        return Segment(
            start=w.start, end=w.end,
            text=f"[CONFLICT]",
            speaker=speaker, source="both",
            text_whisper=w.text,
            text_whisperx=wx.text,
            words=wx.words or w.words,
            extra={**w.extra, **wx.extra},
        )
