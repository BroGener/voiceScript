"""
main.py — CLI entry point.

Usage:
  python main.py                          # prompt for audio path
  python main.py D:/audio.mp3            # specify audio directly
  python main.py D:/audio.mp3 --mode whisper
  python main.py D:/audio.mp3 --mode whisperx
  python main.py D:/audio.mp3 --cut-silence
  python main.py D:/audio.mp3 --speakers Alice Bob

Web UI:
  python main.py --web                   # launch Gradio interface
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import patches
patches.apply_all()

from config import cfg


# ---------------------------------------------------------------------------
# Version check — only on first run (no snapshot yet)
# ---------------------------------------------------------------------------

def _maybe_version_check() -> None:
    """Run version check only if no snapshot exists yet (i.e. first run)."""
    snapshot = cfg.paths.version_file
    if snapshot.exists():
        return  # Already checked once — skip silently

    print("📋 First run: checking environment versions ...")
    from version_checker import collect_versions, save_snapshot
    current = collect_versions()
    save_snapshot(current, snapshot)
    print("   Snapshot saved. Future runs will skip this check.\n")


# ---------------------------------------------------------------------------
# Prompt mode
# ---------------------------------------------------------------------------

def _prompt_and_run() -> None:
    """Interactive prompt for users who prefer not to use CLI args."""
    print("\n" + "=" * 55)
    print("  Whisper Suite — Interactive Mode")
    print("=" * 55)

    # Audio path
    while True:
        raw = input("\nAudio file path (drag & drop or paste): ").strip().strip('"')
        if not raw:
            print("  Path cannot be empty.")
            continue
        audio = Path(raw)
        if audio.exists():
            break
        print(f"  ❌ File not found: {audio}")

    # Mode
    print("\nTranscription mode:")
    print("  1) dual     — Whisper + WhisperX + reconcile (default)")
    print("  2) whisper  — Whisper only (no speaker labels)")
    print("  3) whisperx — WhisperX only (with speaker labels)")
    mode_input = input("Choose [1/2/3, default=1]: ").strip()
    mode = {"1": "dual", "2": "whisper", "3": "whisperx"}.get(mode_input, "dual")

    # Speaker filter
    from speaker_manager import SpeakerManager
    mgr = SpeakerManager()
    known = mgr.list_speakers()
    speaker_filter = None
    if known:
        print(f"\nKnown speakers: {known}")
        raw_filter = input(
            "Restrict matching to specific speakers? (comma-separated, blank=all): "
        ).strip()
        if raw_filter:
            speaker_filter = [s.strip() for s in raw_filter.split(",") if s.strip()]

    # Silence cutting
    cut = input("\nCut silence before transcribing? [y/N]: ").strip().lower() == "y"

    _run(audio, mode=mode, cut_silence=cut, speaker_filter=speaker_filter)


# ---------------------------------------------------------------------------
# Shared run
# ---------------------------------------------------------------------------

def _run(
    audio_path: Path,
    mode: str = "dual",
    cut_silence: bool = False,
    speaker_filter: list[str] | None = None,
) -> None:
    from pipeline import run_pipeline

    print(f"\n{'='*55}")
    print(f"  Audio  : {audio_path.name}")
    print(f"  Mode   : {mode}")
    print(f"  Device : {cfg.device.device.upper()}")
    if speaker_filter:
        print(f"  Filter : {speaker_filter}")
    if cut_silence:
        print(f"  Silence: will cut (remap mode)")
    print(f"{'='*55}")

    result = run_pipeline(
        audio_path,
        mode=mode,
        cut_silence=cut_silence,
        speaker_filter=speaker_filter,
        log=print,
    )

    print(f"\n{'='*55}")
    print(f"  ✅  Complete")
    print(f"  SRT  → {result['output_srt']}")
    print(f"  JSON → {result['output_json']}")

    if result.get("pending_mapping"):
        pm = result["pending_mapping"]
        print(f"\n  ⚠️  Cold-start: open and fill in speaker names:")
        print(f"     {pm}")
        print(f"  Then run: python speaker_setup.py apply --mapping {pm.name}")

    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite — transcription & speaker diarization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio", nargs="?", default=None,
                        help="Audio file path (omit to enter interactively)")
    parser.add_argument("--mode", choices=["dual", "whisper", "whisperx"],
                        default="dual")
    parser.add_argument("--cut-silence", action="store_true",
                        help="Cut silence (remap mode, no re-transcription)")
    parser.add_argument("--speakers", nargs="*", default=None,
                        help="Restrict speaker matching to these names")
    parser.add_argument("--web", action="store_true",
                        help="Launch Gradio web interface")
    args = parser.parse_args()

    _maybe_version_check()

    if args.web:
        from gradio_app import launch
        launch()
        return

    if args.audio is None:
        # No path given — interactive prompt
        _prompt_and_run()
        return

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)

    _run(audio_path, mode=args.mode,
         cut_silence=args.cut_silence,
         speaker_filter=args.speakers or None)


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
