"""
main.py
=======
主入口 — 编排完整转写 pipeline。

用法：
  python main.py                               # 使用 config.py 中的默认音频路径
  python main.py D:/path/to/audio.mp3          # 指定音频路径
  python main.py D:/audio.mp3 --only-whisper   # 只跑 Whisper（无说话人）
  python main.py D:/audio.mp3 --only-whisperx  # 只跑 WhisperX（含说话人）
  python main.py D:/audio.mp3 --skip-version-check

Pipeline（默认双模型）：
  1. 版本检查（可跳过）
  2. Whisper 转写 → 保存单独结果
  3. WhisperX 转写 → 保存单独结果
  4. 声纹提取 & 保存
  5. Reconciler 校对合并 → 保存主要结果

静音裁剪（独立功能，不走此文件）：
  python audio_processor.py D:/audio.mp3
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

# 优先应用补丁（在所有其他导入之前）
import patches
patches.apply_all()

from config import cfg
from reconciler import Reconciler
from speaker_manager import SpeakerManager
from transcriber_whisper import WhisperTranscriber
from transcriber_whisperx import WhisperXTranscriber
from utils import derive_output_paths, save_json, save_srt, save_txt


def run_pipeline(
    audio_path: Path,
    only_whisper: bool = False,
    only_whisperx: bool = False,
) -> None:
    print("\n" + "=" * 62)
    print(f"  Whisper Suite")
    print(f"  音频: {audio_path.name}")
    print(f"  设备: {cfg.device.device.upper()}")
    print("=" * 62)

    out_dir     = cfg.paths.transcripts_dir
    speaker_mgr = SpeakerManager()

    # ── 仅 Whisper ────────────────────────────────────────────
    if only_whisper:
        t = WhisperTranscriber()
        segs = t.transcribe(audio_path)
        t.unload()
        p = derive_output_paths(audio_path, out_dir, "_whisper")
        save_srt(segs, p["srt"])
        save_txt(segs, p["txt"])
        save_json(segs, p["json"])
        print("\n✅ Whisper 转写完成。")
        return

    # ── 仅 WhisperX ───────────────────────────────────────────
    if only_whisperx:
        t = WhisperXTranscriber()
        segs = t.transcribe(audio_path, known_speakers=speaker_mgr.known_speakers() or None)
        _save_speaker_embeddings(t, audio_path, speaker_mgr)
        t.unload()
        p = derive_output_paths(audio_path, out_dir, "_whisperx")
        save_srt(segs, p["srt"])
        save_txt(segs, p["txt"])
        save_json(segs, p["json"])
        print("\n✅ WhisperX 转写完成。")
        return

    # ── 双模型 + Reconciler（默认）───────────────────────────
    # Step A: Whisper
    wt   = WhisperTranscriber()
    w_segs = wt.transcribe(audio_path)
    wt.unload()
    wp = derive_output_paths(audio_path, out_dir, "_whisper")
    save_srt(w_segs, wp["srt"])
    save_txt(w_segs, wp["txt"])
    save_json(w_segs, wp["json"])

    # Step B: WhisperX
    wxt    = WhisperXTranscriber()
    wx_segs = wxt.transcribe(audio_path, known_speakers=speaker_mgr.known_speakers() or None)
    wxp = derive_output_paths(audio_path, out_dir, "_whisperx")
    save_srt(wx_segs, wxp["srt"])
    save_txt(wx_segs, wxp["txt"])
    save_json(wx_segs, wxp["json"])

    # Step C: 声纹提取
    _save_speaker_embeddings(wxt, audio_path, speaker_mgr)
    wxt.unload()

    # Step D: 校对合并
    merged = Reconciler().reconcile(w_segs, wx_segs)
    mp = derive_output_paths(audio_path, out_dir, "_reconciled")
    save_srt(merged, mp["srt"])
    save_txt(merged, mp["txt"])
    save_json(merged, mp["json"])

    print("\n" + "=" * 62)
    print("  ✅ 全部完成！")
    print(f"  📄 主要结果 : {mp['srt'].name}")
    print(f"  📄 主要结果 : {mp['txt'].name}")
    print(f"  📄 Whisper  : {wp['srt'].name}")
    print(f"  📄 WhisperX : {wxp['srt'].name}")
    print(f"  📂 输出目录 : {out_dir}")
    print("=" * 62)
    gc.collect()


def _save_speaker_embeddings(
    wxt: WhisperXTranscriber,
    audio_path: Path,
    speaker_mgr: SpeakerManager,
) -> None:
    try:
        records = wxt.get_diarize_records(audio_path)
        if records:
            speaker_mgr.extract_and_save(audio_path, records)
    except Exception as ex:
        print(f"[main] ⚠️ 声纹提取跳过: {ex}")


def run_version_check(strict: bool = False) -> None:
    from version_checker import collect_versions, compare_snapshots, load_snapshot, save_snapshot
    snapshot_path = cfg.paths.version_file
    current = collect_versions()
    saved   = load_snapshot(snapshot_path)
    if saved is None:
        print("\n⚠️  未找到版本快照，正在生成...")
        save_snapshot(current, snapshot_path)
        print("💡 已生成版本快照，下次运行将自动对比。\n")
    else:
        ok = compare_snapshots(current, saved, strict=strict)
        if not ok:
            print("\n⚠️  如果运行正常可忽略差异。更新快照: python version_checker.py\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite — 双模型语音转写 & 说话人识别",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py D:/audio.mp3
  python main.py D:/audio.mp3 --only-whisper
  python main.py D:/audio.mp3 --only-whisperx
  python main.py D:/audio.mp3 --skip-version-check

静音裁剪（独立）:
  python audio_processor.py D:/audio.mp3
  python audio_processor.py D:/audio.mp3 --remap

人工校正（独立）:
  python correction_tool.py D:/audio.mp3
        """,
    )
    parser.add_argument("audio",                nargs="?", default=None, help="音频文件路径")
    parser.add_argument("--only-whisper",        action="store_true")
    parser.add_argument("--only-whisperx",       action="store_true")
    parser.add_argument("--skip-version-check",  action="store_true")
    parser.add_argument("--strict-version",      action="store_true")
    args = parser.parse_args()

    audio_path = Path(args.audio) if args.audio else cfg.audio_path
    if not audio_path.exists():
        print(f"❌ 音频文件不存在: {audio_path}")
        sys.exit(1)

    if not args.skip_version_check:
        run_version_check(strict=args.strict_version)

    run_pipeline(
        audio_path=audio_path,
        only_whisper=args.only_whisper,
        only_whisperx=args.only_whisperx,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⛔ 用户中断。")
        sys.exit(0)
    except Exception as ex:
        import traceback
        print(f"\n❌ 程序出错: {ex}")
        traceback.print_exc()
        sys.exit(1)
