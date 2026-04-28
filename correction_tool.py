"""
correction_tool.py
==================
Manual correction workflow: compare original vs edited SRT, update voiceprint DB.

Workflow:
  1. Open <stem>_reconciled.srt in a text editor
  2. Change speaker names in parentheses where wrong (e.g. (SPEAKER_00) → (Alice))
  3. Save the file
  4. Run this tool:

  python correction_tool.py D:/audio.mp3
      Auto-finds matching _reconciled.srt and _reconciled.json

  python correction_tool.py D:/audio.mp3 --confident
      Mark corrections as high-confidence (clean recording)
      → embeddings added directly to main pool with weight 0.8

  python correction_tool.py D:/audio.mp3 \\
      --original transcripts/audio_reconciled.json \\
      --corrected transcripts/audio_reconciled.srt
      Specify files explicitly.
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


# ---------------------------------------------------------------------------
# Parse speaker labels from SRT
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(r"\(([^)]+)\)")


def parse_srt_speakers(srt_path: Path) -> list[str | None]:
    """
    Extract speaker label from each subtitle block in the SRT.
    Returns a list aligned with the segment order in the JSON.
    One entry per segment — None if no speaker label found on that block.
    """
    speakers: list[str | None] = []
    current_speaker: str | None = None

    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # SRT block: index line (digit only)
        if line.isdigit():
            # Next line is the timestamp, skip it
            i += 2
            current_speaker = None
            # Read content lines until blank
            while i < len(lines) and lines[i].strip():
                content = lines[i].strip()
                m = _SPEAKER_RE.search(content)
                if m:
                    label = m.group(1).strip()
                    # Ignore meta-labels inserted by the system
                    if not label.startswith("["):
                        current_speaker = label
                        break
                i += 1
            speakers.append(current_speaker)
        i += 1

    return speakers


def find_files(audio_path: Path) -> tuple[Path, Path]:
    """Auto-find matching JSON and SRT for a given audio file."""
    trans_dir = cfg.paths.transcripts_dir
    stem = audio_path.stem

    for suffix in ["_reconciled", "_whisperx", ""]:
        json_path = trans_dir / f"{stem}{suffix}.json"
        srt_path  = trans_dir / f"{stem}{suffix}.srt"
        if json_path.exists() and srt_path.exists():
            return json_path, srt_path

    raise FileNotFoundError(
        f"No matching transcription files found for {stem} in {trans_dir}"
    )


# ---------------------------------------------------------------------------
# Main correction logic
# ---------------------------------------------------------------------------

def run_correction(
    audio_path: Path,
    original_json: Path,
    corrected_srt: Path,
    confident: bool = False,
) -> None:
    print(f"\n[Correction] Loading original: {original_json.name}")
    original_segs = load_json(original_json)

    print(f"[Correction] Parsing edited SRT: {corrected_srt.name}")
    corrected_speakers = parse_srt_speakers(corrected_srt)

    if len(corrected_speakers) != len(original_segs):
        print(
            f"⚠️  Segment count mismatch: JSON={len(original_segs)}, "
            f"SRT={len(corrected_speakers)}\n"
            f"   Ensure the SRT has not had blocks added or removed — "
            f"only speaker names should be changed."
        )
        n = min(len(corrected_speakers), len(original_segs))
        original_segs      = original_segs[:n]
        corrected_speakers = corrected_speakers[:n]

    # Build corrected segment list
    corrected_segs = [
        Segment(
            start=seg.start, end=seg.end, text=seg.text,
            speaker=new_sp or seg.speaker,
            source=seg.source,
            text_whisper=seg.text_whisper,
            text_whisperx=seg.text_whisperx,
            words=seg.words, extra=seg.extra,
        )
        for seg, new_sp in zip(original_segs, corrected_speakers)
    ]

    diffs = [
        (orig, corr) for orig, corr in zip(original_segs, corrected_segs)
        if orig.speaker != corr.speaker
    ]

    if not diffs:
        print("\n✅ No speaker changes detected — nothing to correct.")
        return

    print(f"\n[Correction] {len(diffs)} speaker change(s) found:")
    for orig, corr in diffs:
        print(f"  [{orig.start:.1f}s]  {orig.speaker} → {corr.speaker}"
              f"  |  {orig.text[:50]}")

    mode = "confident (high weight)" if confident else "auto (distance-based)"
    print(f"\n  Mode: {mode}")
    confirm = input("\nApply corrections and update voiceprint database? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Cancelled.")
        return

    mgr = SpeakerManager()
    mgr.apply_corrections_from_srt_diff(
        audio_path, corrected_segs, original_segs, confident=confident
    )
    print("\n✅ Voiceprint database updated.")
    print("   Future transcriptions will use the updated speaker models.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite — Manual speaker correction tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("audio",       type=str,
                        help="Path to the original audio file")
    parser.add_argument("--original",  type=str, default=None,
                        help="Original JSON transcript (auto-found if omitted)")
    parser.add_argument("--corrected", type=str, default=None,
                        help="Edited SRT file (auto-found if omitted)")
    parser.add_argument("--confident", action="store_true",
                        help="Mark as high-confidence correction (clean recording)")

    args = parser.parse_args()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    if args.original and args.corrected:
        json_path = Path(args.original)
        srt_path  = Path(args.corrected)
    else:
        try:
            json_path, srt_path = find_files(audio_path)
        except FileNotFoundError as ex:
            print(f"❌ {ex}")
            sys.exit(1)

    try:
        run_correction(audio_path, json_path, srt_path, confident=args.confident)
    except KeyboardInterrupt:
        print("\n⛔ Interrupted.")
        sys.exit(0)
    except Exception as ex:
        import traceback
        print(f"\n❌ Error: {ex}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
