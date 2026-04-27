"""
correction_tool.py
==================
【独立工具】人工校正辅助脚本。

工作流：
  1. 转写完成后，用文本编辑器打开 <stem>_reconciled.txt
  2. 将错误的说话人标签改成正确的（只改括号里的名字，格式不变）
  3. 运行此脚本，它会对比原始 JSON 和修改后的 TXT，找出差异
  4. 自动更新声纹数据库，下次识别会更准

用法：
  python correction_tool.py D:/audio.mp3
      自动查找 transcripts/ 目录下的 _reconciled.json 和 _reconciled.txt

  python correction_tool.py D:/audio.mp3 \\
      --original D:/whisper_data/transcripts/audio_reconciled.json \\
      --corrected D:/whisper_data/transcripts/audio_reconciled.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import patches
patches.apply_all()

from config import cfg
from speaker_manager import SpeakerManager
from utils import Segment, load_json


_SPEAKER_RE = re.compile(r"\(([^)]+)\)")


def parse_txt_speakers(txt_path: Path) -> list[str | None]:
    """从 TXT 文件中按行解析说话人标签，返回与 segments 顺序对应的列表。"""
    speakers = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _SPEAKER_RE.search(line)
            if m:
                speakers.append(m.group(1).strip())
    return speakers


def find_output_files(audio_path: Path) -> tuple[Path, Path]:
    trans_dir = cfg.paths.transcripts_dir
    stem = audio_path.stem
    for suffix in ["_reconciled", "_whisperx", ""]:
        json_path = trans_dir / f"{stem}{suffix}.json"
        txt_path  = trans_dir / f"{stem}{suffix}.txt"
        if json_path.exists() and txt_path.exists():
            return json_path, txt_path
    raise FileNotFoundError(
        f"未找到转写文件，请确认 {trans_dir} 中有 {stem}_reconciled.json/.txt"
    )


def run_correction(audio_path: Path, original_json: Path, corrected_txt: Path) -> None:
    print(f"\n[校正工具] 加载原始转写: {original_json.name}")
    original_segs = load_json(original_json)

    print(f"[校正工具] 加载修改后的 TXT: {corrected_txt.name}")
    corrected_speakers = parse_txt_speakers(corrected_txt)

    if len(corrected_speakers) != len(original_segs):
        print(
            f"⚠️  片段数量不匹配: JSON={len(original_segs)}, TXT={len(corrected_speakers)}\n"
            f"   请确保 TXT 文件没有增删行，只改了说话人名称。"
        )
        n = min(len(corrected_speakers), len(original_segs))
        original_segs      = original_segs[:n]
        corrected_speakers = corrected_speakers[:n]

    corrected_segs = [
        Segment(
            start=seg.start, end=seg.end, text=seg.text,
            speaker=new_sp or seg.speaker,
            source=seg.source,
            text_whisper=seg.text_whisper, text_whisperx=seg.text_whisperx,
            words=seg.words, extra=seg.extra,
        )
        for seg, new_sp in zip(original_segs, corrected_speakers)
    ]

    diffs = [
        (orig, corr) for orig, corr in zip(original_segs, corrected_segs)
        if orig.speaker != corr.speaker
    ]

    if not diffs:
        print("\n✅ 未发现说话人差异，无需校正。")
        return

    print(f"\n[校正工具] 发现 {len(diffs)} 处说话人变更：")
    for orig, corr in diffs:
        print(f"  [{orig.start:.1f}s] {orig.speaker} → {corr.speaker} | {orig.text[:50]}")

    confirm = input("\n确认进行校正并更新声纹数据库？[y/N] ").strip().lower()
    if confirm != "y":
        print("已取消。")
        return

    mgr = SpeakerManager()
    mgr.apply_corrections_from_file(audio_path, corrected_segs, original_segs)
    print("\n✅ 声纹数据库已更新，下次转写将使用更准确的说话人识别。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite 人工校正工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
校正步骤:
  1. 用文本编辑器打开 <stem>_reconciled.txt
  2. 找到说话人标签错误的行（格式: 时间戳  (说话人): 文本）
  3. 将括号内的名字改为正确的说话人名称
  4. 保存后运行此脚本
        """,
    )
    parser.add_argument("audio",       type=str, help="原始音频文件路径")
    parser.add_argument("--original",  type=str, default=None, help="原始转写 JSON 路径（可选）")
    parser.add_argument("--corrected", type=str, default=None, help="修改后的 TXT 路径（可选）")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ 音频文件不存在: {audio_path}")
        sys.exit(1)

    if args.original and args.corrected:
        json_path = Path(args.original)
        txt_path  = Path(args.corrected)
    else:
        try:
            json_path, txt_path = find_output_files(audio_path)
        except FileNotFoundError as ex:
            print(f"❌ {ex}")
            sys.exit(1)

    run_correction(audio_path, json_path, txt_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⛔ 已中断。")
        sys.exit(0)
    except Exception as ex:
        import traceback
        print(f"\n❌ 出错: {ex}")
        traceback.print_exc()
        sys.exit(1)
