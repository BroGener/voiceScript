"""
speaker_setup.py
================
Cold-start and speaker management utility.

Commands:
  python speaker_setup.py status
      Print current speaker database summary.

  python speaker_setup.py apply [--audio AUDIO_PATH]
      Read pending_mapping.json, apply speaker names:
        - renames labels in the SRT file
        - extracts embeddings and seeds the database
        - optionally adds as anchors (gold standard)

  python speaker_setup.py anchor --name NAME --audio AUDIO_PATH --start S --end E
      Manually add a gold-standard anchor from a specific audio segment.

  python speaker_setup.py weight --file FILENAME --weight 0.5
      Update file_weight for all embeddings from a source file.

  python speaker_setup.py drop --file FILENAME
      Permanently remove all embeddings from a source file.

  python speaker_setup.py rename --old OLD_NAME --new NEW_NAME
      Rename a speaker in the database and update all SRT files.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import patches
patches.apply_all()

from config import cfg
from speaker_manager import SpeakerManager
from utils import load_json, save_srt, save_json


# ---------------------------------------------------------------------------
# apply: read pending_mapping.json → rename SRT → seed database
# ---------------------------------------------------------------------------

def cmd_apply(args) -> None:
    mapping_path = cfg.paths.pending_mapping_file
    if not mapping_path.exists():
        print(f"❌ No pending mapping found at: {mapping_path}")
        print("   Run main.py first to generate a cold-start transcription.")
        sys.exit(1)

    mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_file  = mapping_data["source_file"]
    speakers     = mapping_data["speakers"]

    # Validate that all names are filled in
    unfilled = [lbl for lbl, v in speakers.items() if not v.get("real_name", "").strip()]
    if unfilled:
        print(f"❌ The following labels have no real_name assigned: {unfilled}")
        print(f"   Edit {mapping_path} and fill in the 'real_name' fields.")
        sys.exit(1)

    label_to_name = {lbl: v["real_name"].strip() for lbl, v in speakers.items()}
    print(f"\n[Setup] Applying mapping: {label_to_name}")

    # Find SRT file for this source
    srt_candidates = list(cfg.paths.transcripts_dir.glob(
        f"{Path(source_file).stem}*.srt"
    ))
    if not srt_candidates:
        print(f"⚠️  No SRT file found for {source_file} — skipping SRT rename.")
    else:
        for srt_path in srt_candidates:
            _rename_labels_in_srt(srt_path, label_to_name)

    # Find JSON and update speaker labels
    json_candidates = list(cfg.paths.transcripts_dir.glob(
        f"{Path(source_file).stem}*.json"
    ))
    for json_path in json_candidates:
        _rename_labels_in_json(json_path, label_to_name)

    # Determine audio path
    audio_path = Path(args.audio) if args.audio else _find_audio(source_file)
    if audio_path is None or not audio_path.exists():
        print(f"⚠️  Audio file not found — skipping embedding extraction.")
        print(f"   Re-run with: python speaker_setup.py apply --audio PATH_TO_{source_file}")
    else:
        _seed_database(audio_path, mapping_data, label_to_name, as_anchor=args.anchor)

    # Clean up
    mapping_path.unlink(missing_ok=True)
    print("\n✅ Mapping applied. Database seeded.")
    print("   Next run of main.py will use Route B (constrained clustering).")


def _rename_labels_in_srt(srt_path: Path, label_to_name: dict[str, str]) -> None:
    text = srt_path.read_text(encoding="utf-8")
    for label, name in label_to_name.items():
        text = text.replace(f"({label})", f"({name})")
    srt_path.write_text(text, encoding="utf-8")
    print(f"  ✏️  Updated: {srt_path.name}")


def _rename_labels_in_json(json_path: Path, label_to_name: dict[str, str]) -> None:
    try:
        segs = load_json(json_path)
        for seg in segs:
            if seg.speaker in label_to_name:
                seg.speaker = label_to_name[seg.speaker]
        save_json(segs, json_path)
        print(f"  ✏️  Updated: {json_path.name}")
    except Exception as ex:
        print(f"  ⚠️  Could not update {json_path.name}: {ex}")


def _seed_database(
    audio_path: Path,
    mapping_data: dict,
    label_to_name: dict[str, str],
    as_anchor: bool = False,
) -> None:
    """Extract embeddings from the audio and seed the speaker database."""
    from pyannote.audio import Model, Inference
    import torchaudio

    print(f"\n[Setup] Extracting embeddings from: {audio_path.name}")
    mgr = SpeakerManager()

    model     = Model.from_pretrained("pyannote/embedding",
                                      use_auth_token=cfg.hf_token)
    inference = Inference(model, window="whole")
    waveform, sr = torchaudio.load(str(audio_path))

    # Reconstruct segment list from sample_segments timestamps
    # (approximate — just for seeding, not precision-critical)
    for label, info in mapping_data["speakers"].items():
        name = label_to_name[label]
        embeddings = []

        for ts_str in info.get("sample_segments", []):
            # Parse "HH:MM:SS,mmm --> HH:MM:SS,mmm"
            m = re.match(
                r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)",
                ts_str
            )
            if not m:
                continue
            g    = [int(x) for x in m.groups()]
            start = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
            end   = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
            dur   = end - start

            if dur < cfg.speaker.min_segment_duration_s:
                continue

            s     = int(start * sr)
            e     = int(end   * sr)
            chunk = waveform[:, s:e]
            try:
                emb = inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr})
                import numpy as np
                embeddings.append(np.array(emb))
            except Exception as ex:
                print(f"  ⚠️  Embedding failed for {ts_str}: {ex}")

        if embeddings:
            if as_anchor:
                for emb in embeddings:
                    mgr.add_anchor(name, emb, source_file=audio_path.name)
            else:
                mgr.store_embeddings(name, embeddings,
                                     source_file=audio_path.name,
                                     file_weight=1.0)
            print(f"  ✅ {name}: {len(embeddings)} embedding(s) stored "
                  f"({'anchor' if as_anchor else 'auto'})")
        else:
            print(f"  ⚠️  {name}: no valid embeddings extracted.")


def _find_audio(source_file: str) -> Path | None:
    """Try to find the audio file in common locations."""
    candidates = [
        cfg.audio_path.parent / source_file,
        Path(source_file),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# anchor: add gold-standard anchor from specific timestamp
# ---------------------------------------------------------------------------

def cmd_anchor(args) -> None:
    import numpy as np
    from pyannote.audio import Model, Inference
    import torchaudio

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    model     = Model.from_pretrained("pyannote/embedding",
                                      use_auth_token=cfg.hf_token)
    inference = Inference(model, window="whole")
    waveform, sr = torchaudio.load(str(audio_path))

    start_s = float(args.start)
    end_s   = float(args.end)
    chunk   = waveform[:, int(start_s * sr): int(end_s * sr)]
    emb     = np.array(inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr}))

    mgr = SpeakerManager()
    mgr.add_anchor(args.name, emb, source_file=audio_path.name)
    print(f"✅ Anchor added for '{args.name}' from {audio_path.name} "
          f"[{start_s:.1f}s – {end_s:.1f}s]")


# ---------------------------------------------------------------------------
# weight / drop / rename / status
# ---------------------------------------------------------------------------

def cmd_weight(args) -> None:
    mgr = SpeakerManager()
    mgr.set_file_weight(args.file, float(args.weight))


def cmd_drop(args) -> None:
    confirm = input(f"Permanently remove all embeddings from '{args.file}'? [y/N] ")
    if confirm.strip().lower() != "y":
        print("Cancelled.")
        return
    mgr = SpeakerManager()
    mgr.drop_file_embeddings(args.file)


def cmd_rename(args) -> None:
    mgr = SpeakerManager()
    if args.old not in mgr.list_speakers():
        print(f"❌ Speaker '{args.old}' not found in database.")
        sys.exit(1)

    # Rename JSON file
    speakers_dir = cfg.paths.speakers_dir
    old_file = speakers_dir / f"{args.old.replace(' ', '_')}.json"
    new_file = speakers_dir / f"{args.new.replace(' ', '_')}.json"

    data = json.loads(old_file.read_text(encoding="utf-8"))
    data["name"] = args.new
    new_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    old_file.unlink()

    # Update all SRT and JSON transcript files
    for srt_path in cfg.paths.transcripts_dir.glob("*.srt"):
        text = srt_path.read_text(encoding="utf-8")
        if f"({args.old})" in text:
            srt_path.write_text(text.replace(f"({args.old})", f"({args.new})"),
                                 encoding="utf-8")
            print(f"  ✏️  {srt_path.name}")

    for json_path in cfg.paths.transcripts_dir.glob("*.json"):
        try:
            segs = load_json(json_path)
            changed = False
            for seg in segs:
                if seg.speaker == args.old:
                    seg.speaker = args.new
                    changed = True
            if changed:
                save_json(segs, json_path)
                print(f"  ✏️  {json_path.name}")
        except Exception:
            pass

    print(f"✅ Renamed '{args.old}' → '{args.new}' in database and all transcripts.")


def cmd_status(args) -> None:
    mgr = SpeakerManager()
    mgr.print_db_summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite — Speaker database setup & management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # apply
    p_apply = sub.add_parser("apply", help="Apply pending_mapping.json")
    p_apply.add_argument("--audio",  type=str, default=None,
                         help="Path to the source audio file")
    p_apply.add_argument("--anchor", action="store_true",
                         help="Store embeddings as gold-standard anchors")

    # anchor
    p_anchor = sub.add_parser("anchor", help="Add a gold-standard anchor")
    p_anchor.add_argument("--name",  required=True)
    p_anchor.add_argument("--audio", required=True)
    p_anchor.add_argument("--start", required=True, type=float,
                          help="Segment start (seconds)")
    p_anchor.add_argument("--end",   required=True, type=float,
                          help="Segment end (seconds)")

    # weight
    p_weight = sub.add_parser("weight", help="Set file weight for all embeddings")
    p_weight.add_argument("--file",   required=True, help="Source filename (e.g. A.mp3)")
    p_weight.add_argument("--weight", required=True, help="Weight 0.0–1.0")

    # drop
    p_drop = sub.add_parser("drop", help="Remove all embeddings from a file")
    p_drop.add_argument("--file", required=True)

    # rename
    p_rename = sub.add_parser("rename", help="Rename a speaker")
    p_rename.add_argument("--old", required=True)
    p_rename.add_argument("--new", required=True)

    # status
    sub.add_parser("status", help="Print speaker database summary")

    args = parser.parse_args()

    dispatch = {
        "apply":  cmd_apply,
        "anchor": cmd_anchor,
        "weight": cmd_weight,
        "drop":   cmd_drop,
        "rename": cmd_rename,
        "status": cmd_status,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    try:
        dispatch[args.command](args)
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
