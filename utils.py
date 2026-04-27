"""
utils.py
========
公共工具：Segment 数据结构、时间格式化、SRT/TXT/JSON 文件输出。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Segment:
    """一条转写片段，携带来源标记和完整元数据。"""
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_UNKNOWN"
    source: Literal["whisper", "whisperx", "reconciled", "both"] = "reconciled"
    # 当 source=="both" 时保留两侧原文
    text_whisper: str = ""
    text_whisperx: str = ""
    # 词级对齐（whisperx align 返回的 words 列表）
    words: list = field(default_factory=list)
    # 原始置信度等扩展字段（方便后期训练）
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ── 时间格式化 ───────────────────────────────────────────────

def fmt_srt(seconds: float) -> str:
    """秒 → SRT 格式  HH:MM:SS,mmm"""
    ms = int(round((seconds % 1) * 1000))
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_bracket(seconds: float) -> str:
    """秒 → 方括号格式  [HH:MM:SS]"""
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"[{h:02d}:{m:02d}:{s:02d}]"


# ── 文件输出 ─────────────────────────────────────────────────

def save_srt(segments: list[Segment], path: Path) -> None:
    """保存 SRT 字幕文件。冲突句用 [WHISPER] / [WHISPERX] 双行输出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            time_line = f"{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}"
            if seg.source == "both":
                f.write(
                    f"{idx}\n{time_line}\n"
                    f"  ({seg.speaker}) [WHISPER]  : {seg.text_whisper}\n"
                    f"  ({seg.speaker}) [WHISPERX] : {seg.text_whisperx}\n\n"
                )
            else:
                src_tag = f"[{seg.source.upper()}] " if seg.source != "reconciled" else ""
                f.write(f"{idx}\n{time_line}\n  ({seg.speaker}) {src_tag}{seg.text}\n\n")
    print(f"  💾 SRT  → {path}")


def save_txt(segments: list[Segment], path: Path) -> None:
    """保存纯文本转写，带时间轴和说话人标签。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            ts = f"{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}"
            if seg.source == "both":
                f.write(
                    f"{ts}  ({seg.speaker})\n"
                    f"  [WHISPER]  : {seg.text_whisper}\n"
                    f"  [WHISPERX] : {seg.text_whisperx}\n"
                )
            else:
                src_tag = f"[{seg.source.upper()}] " if seg.source != "reconciled" else ""
                f.write(f"{ts}  ({seg.speaker}) {src_tag}{seg.text}\n")
    print(f"  💾 TXT  → {path}")


def save_json(segments: list[Segment], path: Path) -> None:
    """保存完整 JSON（含词级对齐、extra 字段），用于后期数据调取和训练。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)
    print(f"  💾 JSON → {path}")


def load_json(path: Path) -> list[Segment]:
    """从 JSON 文件重新加载 Segment 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Segment(**d) for d in data]


def derive_output_paths(audio_path: Path, out_dir: Path, suffix: str = "") -> dict[str, Path]:
    """根据音频文件名推导输出文件路径。"""
    stem = audio_path.stem + suffix
    return {
        "srt":  out_dir / f"{stem}.srt",
        "txt":  out_dir / f"{stem}.txt",
        "json": out_dir / f"{stem}.json",
    }
