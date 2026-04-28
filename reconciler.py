"""
reconciler.py
=============
双模型结果校对合并器。

策略：
  - 时间轴以 Whisper 为基准（更贴近真实时间）
  - 说话人标签始终来自 WhisperX（Whisper 无此信息）
  - 文本比较前先做归一化（去标点/空白），避免 ... 和空格造成假冲突
  - 支持多对一合并：Whisper 把一句拆成多段，WhisperX 是整句的情况

断句不一致的处理：
  Whisper 断句更细，WhisperX 断句更粗是常见情况。
  在时间容差内，如果多个 Whisper 片段的文本拼起来与一个 WhisperX 片段接近，
  则把这组 Whisper 片段合并为一个条目再做比对。
  不需要语义分析，纯时间窗口 + 文本归一化即可解决大多数情况。
"""

from __future__ import annotations
import re
from config import cfg
from utils import Segment


# ── 文本归一化 ───────────────────────────────────────────────

# 去掉的字符：省略号、连字符（用于断句）、多余标点和空白
_STRIP_RE = re.compile(r"[.。,，!！?？\-–—…\s]+")

def _normalize(text: str) -> str:
    """
    归一化文本用于比较（不改变输出文本，只用于判断是否一致）。
    处理：大小写、省略号、连字符、多余空格、全/半角标点。
    """
    return _STRIP_RE.sub("", text).lower().strip()


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
        used_w:  set[int] = set()
        used_wx: set[int] = set()

        # ── Pass 1: 多对一合并检测 ────────────────────────────
        # 找出 Whisper 中多个相邻片段对应 WhisperX 中单个片段的情况
        multi_groups = self._find_multi_to_one(whisper_segs, whisperx_segs)

        for wx_idx, w_indices in multi_groups.items():
            wx_seg = whisperx_segs[wx_idx]
            w_group = [whisper_segs[i] for i in w_indices]

            # 把这组 Whisper 片段合并成一个虚拟片段
            merged_w = self._merge_whisper_group(w_group)
            merged.append(self._merge_pair(merged_w, wx_seg))

            for i in w_indices:
                used_w.add(i)
            used_wx.add(wx_idx)

        # ── Pass 2: 常规一对一匹配 ───────────────────────────
        for i, w_seg in enumerate(whisper_segs):
            if i in used_w:
                continue
            matches = self._find_matches(w_seg, whisperx_segs, used_wx)
            if not matches:
                merged.append(w_seg)
                continue
            best_idx, wx_seg = matches[0]
            used_wx.add(best_idx)
            used_w.add(i)
            merged.append(self._merge_pair(w_seg, wx_seg))

        # ── Pass 3: WhisperX 独有片段 ────────────────────────
        for i, wx_seg in enumerate(whisperx_segs):
            if i not in used_wx:
                merged.append(wx_seg)

        merged.sort(key=lambda s: s.start)

        conflicts  = sum(1 for s in merged if s.source == "both")
        reconciled = sum(1 for s in merged if s.source == "reconciled")
        w_only     = sum(1 for s in merged if s.source == "whisper")
        wx_only    = sum(1 for s in merged if s.source == "whisperx")
        multi      = len(multi_groups)
        print(f"[Reconciler] 完成: 总计 {len(merged)} 片段 | "
              f"合并 {reconciled} | 冲突 {conflicts} | "
              f"多对一合并 {multi} | Whisper独有 {w_only} | WhisperX独有 {wx_only}")
        return merged

    # ── 多对一：找出 Whisper 多段 → WhisperX 单段的组 ────────

    def _find_multi_to_one(
        self,
        whisper_segs: list[Segment],
        whisperx_segs: list[Segment],
    ) -> dict[int, list[int]]:
        """
        返回 {wx_idx: [w_idx, w_idx, ...]} 的映射，
        表示哪些 WhisperX 单个片段对应多个 Whisper 片段。

        条件：
          1. 多个 Whisper 片段的时间范围完全落在同一个 WhisperX 片段的容差范围内
          2. 这些 Whisper 片段是相邻的（中间没有跳跃）
          3. 拼合后的文本归一化比较一致（相似度 > 阈值）
        """
        tol = self._cfg.time_tolerance_s
        result: dict[int, list[int]] = {}

        for wx_idx, wx_seg in enumerate(whisperx_segs):
            wx_start = wx_seg.start - tol
            wx_end   = wx_seg.end   + tol

            # 找出时间范围内的所有 Whisper 片段
            contained = [
                i for i, w in enumerate(whisper_segs)
                if w.start >= wx_start and w.end <= wx_end
            ]

            if len(contained) < 2:
                continue  # 少于 2 个不需要多对一合并

            # 验证是否相邻（中间没有跳跃）
            contained.sort()
            if contained != list(range(contained[0], contained[-1] + 1)):
                continue

            # 拼合 Whisper 文本，与 WhisperX 归一化比较
            combined_text = " ".join(whisper_segs[i].text.strip() for i in contained)
            if _normalize(combined_text) == _normalize(wx_seg.text):
                result[wx_idx] = contained

        return result

    @staticmethod
    def _merge_whisper_group(segs: list[Segment]) -> Segment:
        """把多个 Whisper 片段合并成一个虚拟片段（用于多对一比对）。"""
        return Segment(
            start=segs[0].start,
            end=segs[-1].end,
            text=" ".join(s.text.strip() for s in segs),
            speaker="SPEAKER_UNKNOWN",
            source="whisper",
            words=[w for s in segs for w in s.words],
            extra={k: v for s in segs for k, v in s.extra.items()},
        )

    # ── 常规一对一匹配 ───────────────────────────────────────

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

        # 归一化比较：去标点、去空白、统一大小写
        if _normalize(text_w) == _normalize(text_wx):
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
            return Segment(start=w.start, end=w.end, text=text_w,
                           speaker=speaker, source="whisper",
                           words=w.words, extra=w.extra)
        if strategy == "prefer_whisperx":
            return Segment(start=w.start, end=w.end, text=text_wx,
                           speaker=speaker, source="whisperx",
                           words=wx.words, extra=wx.extra)
        # keep_both（默认）
        return Segment(
            start=w.start, end=w.end,
            text="[CONFLICT]",
            speaker=speaker, source="both",
            text_whisper=text_w,
            text_whisperx=text_wx,
            words=wx.words or w.words,
            extra={**w.extra, **wx.extra},
        )
