"""
main.py — Main pipeline orchestrator.

Usage:
  python main.py                               # use AUDIO_PATH from config
  python main.py D:/audio.mp3
  python main.py D:/audio.mp3 --only-whisper
  python main.py D:/audio.mp3 --only-whisperx
  python main.py D:/audio.mp3 --skip-version-check

First run (empty database):
  → WhisperX uses DiarizationPipeline (cold-start)
  → pending_mapping.json is written
  → run: python speaker_setup.py apply
  → subsequent runs use Route B (constrained clustering)

Silence cutting (standalone):
  python audio_processor.py D:/audio.mp3
  python audio_processor.py D:/audio.mp3 --remap

Manual correction (standalone):
  python correction_tool.py D:/audio.mp3 [--confident]
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import patches
patches.apply_all()

from config import cfg
from reconciler import Reconciler
from speaker_manager import SpeakerManager
from transcriber_whisper import WhisperTranscriber
from transcriber_whisperx import WhisperXTranscriber
from utils import derive_output_paths, save_json, save_srt


def run_pipeline(
    audio_path: Path,
    only_whisper: bool = False,
    only_whisperx: bool = False,
) -> None:
    print("\n" + "=" * 62)
    print(f"  Whisper Suite")
    print(f"  Audio  : {audio_path.name}")
    print(f"  Device : {cfg.device.device.upper()}")
    print("=" * 62)

    out_dir     = cfg.paths.transcripts_dir
    speaker_mgr = SpeakerManager()

    # ── Whisper only ──────────────────────────────────────────
    if only_whisper:
        t    = WhisperTranscriber()
        segs = t.transcribe(audio_path)
        t.unload()
        p = derive_output_paths(audio_path, out_dir, "_whisper")
        save_srt(segs, p["srt"])
        save_json(segs, p["json"])
        print("\n✅ Whisper transcription complete.")
        return

    # ── WhisperX only ─────────────────────────────────────────
    if only_whisperx:
        t    = WhisperXTranscriber()
        segs = t.transcribe(audio_path, speaker_manager=speaker_mgr)
        t.unload()
        p = derive_output_paths(audio_path, out_dir, "_whisperx")
        save_srt(segs, p["srt"])
        save_json(segs, p["json"])
        print("\n✅ WhisperX transcription complete.")
        return

    # ── Dual model + reconcile (default) ──────────────────────

    # Step A: Whisper
    wt     = WhisperTranscriber()
    w_segs = wt.transcribe(audio_path)
    wt.unload()
    wp = derive_output_paths(audio_path, out_dir, "_whisper")
    save_srt(w_segs, wp["srt"])
    save_json(w_segs, wp["json"])

    # Step B: WhisperX (Route B or cold-start)
    wxt     = WhisperXTranscriber()
    wx_segs = wxt.transcribe(audio_path, speaker_manager=speaker_mgr)
    wxt.unload()
    wxp = derive_output_paths(audio_path, out_dir, "_whisperx")
    save_srt(wx_segs, wxp["srt"])
    save_json(wx_segs, wxp["json"])

    # Step C: Reconcile
    merged = Reconciler().reconcile(w_segs, wx_segs)
    mp = derive_output_paths(audio_path, out_dir, "_reconciled")
    save_srt(merged, mp["srt"])
    save_json(merged, mp["json"])

    gc.collect()

    print("\n" + "=" * 62)
    print("  ✅ Done!")
    print(f"  📄 Main output : {mp['srt'].name}")
    print(f"  📄 Whisper     : {wp['srt'].name}")
    print(f"  📄 WhisperX    : {wxp['srt'].name}")
    print(f"  📂 Output dir  : {out_dir}")

    if (cfg.paths.pending_mapping_file.exists()):
        print(f"\n  ⚠️  Cold-start: speaker names needed.")
        print(f"     Edit: {cfg.paths.pending_mapping_file}")
        print(f"     Then: python speaker_setup.py apply")

    print("=" * 62)


def run_version_check(strict: bool = False) -> None:
    from version_checker import collect_versions, compare_snapshots, load_snapshot, save_snapshot
    snapshot_path = cfg.paths.version_file
    current = collect_versions()
    saved   = load_snapshot(snapshot_path)
    if saved is None:
        print("\n⚠️  No version snapshot found — generating one now...")
        save_snapshot(current, snapshot_path)
        print("   Snapshot saved. Future runs will compare against it.\n")
    else:
        ok = compare_snapshots(current, saved, strict=strict)
        if not ok:
            print("\n⚠️  Differences found. If everything runs fine, ignore them.")
            print("   To update snapshot: python version_checker.py\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite — dual-model transcription & speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio",               nargs="?", default=None)
    parser.add_argument("--only-whisper",       action="store_true")
    parser.add_argument("--only-whisperx",      action="store_true")
    parser.add_argument("--skip-version-check", action="store_true")
    parser.add_argument("--strict-version",     action="store_true")
    args = parser.parse_args()

    audio_path = Path(args.audio) if args.audio else cfg.audio_path
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
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
        print("\n\n⛔ Interrupted.")
        sys.exit(0)
    except Exception as ex:
        import traceback
        print(f"\n❌ Error: {ex}")
        traceback.print_exc()
        sys.exit(1)
