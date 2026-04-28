"""
handlers/speaker_handlers.py
=============================
Business logic for the Speaker DB tab.
"""

from __future__ import annotations
import json
import traceback
from pathlib import Path

from config import cfg
from speaker_manager import SpeakerManager


def db_summary() -> str:
    mgr = SpeakerManager()
    if not mgr.has_speakers():
        return "Database is empty. Run a transcription first."
    lines = ["Speaker database:\n"]
    for name in mgr.list_speakers():
        d = mgr._db[name]
        lines.append(
            f"  {name:<22}  anchors={len(d.get('anchors',[]))}  "
            f"emb={len(d.get('embeddings',[]))}  "
            f"soft={len(d.get('soft_corrections',[]))}  "
            f"corrections={d.get('correction_count',0)}"
        )
    return "\n".join(lines)


def list_pending() -> str:
    files = list(cfg.paths.pending_mapping_dir.glob("pending_*.json"))
    if not files:
        return "No pending mappings."
    return "\n".join(f.name for f in files)


def apply_mapping(mapping_file, audio_file, as_anchor: bool) -> str:
    if mapping_file is None:
        return "❌ Select a pending_*.json file."
    mp = Path(mapping_file if isinstance(mapping_file, str) else mapping_file.name)
    try:
        data     = json.loads(mp.read_text(encoding="utf-8"))
        speakers = data.get("speakers", {})
        unfilled = [k for k, v in speakers.items() if not v.get("real_name", "").strip()]
        if unfilled:
            return f"❌ Fill in real_name for: {unfilled}"

        label_to_name = {k: v["real_name"].strip() for k, v in speakers.items()}
        source_file   = data.get("source_file", "")

        from speaker_setup import _rename_labels_in_srt, _rename_labels_in_json
        updated = []
        for p in cfg.paths.transcripts_dir.glob(f"{Path(source_file).stem}*.srt"):
            _rename_labels_in_srt(p, label_to_name)
            updated.append(p.name)
        for p in cfg.paths.transcripts_dir.glob(f"{Path(source_file).stem}*.json"):
            _rename_labels_in_json(p, label_to_name)
            updated.append(p.name)

        db_msg = ""
        if audio_file is not None:
            ap = Path(audio_file if isinstance(audio_file, str) else audio_file.name)
            from speaker_setup import _seed_database
            _seed_database(ap, data, label_to_name, as_anchor=as_anchor)
            db_msg = f"\nDB seeded from {ap.name}."

        mp.unlink(missing_ok=True)
        return f"✅ Applied: {label_to_name}\nUpdated: {', '.join(updated)}{db_msg}"
    except Exception:
        return f"❌ {traceback.format_exc()}"


def set_weight(filename: str, weight: float) -> str:
    if not filename.strip():
        return "❌ Enter filename."
    SpeakerManager().set_file_weight(filename.strip(), float(weight))
    return f"✅ Weight={weight} for '{filename}'."


def drop_file(filename: str) -> str:
    if not filename.strip():
        return "❌ Enter filename."
    SpeakerManager().drop_file_embeddings(filename.strip())
    return f"✅ Removed all embeddings from '{filename}'."


def rename_speaker(old: str, new: str) -> str:
    if not old.strip() or not new.strip():
        return "❌ Both names required."
    from speaker_setup import cmd_rename
    class _A:
        old_name = old.strip()
        new_name = new.strip()
    # cmd_rename expects args.old / args.new
    class _Args:
        pass
    a = _Args()
    a.old = old.strip()
    a.new = new.strip()
    try:
        cmd_rename(a)
        return f"✅ Renamed '{old}' → '{new}'."
    except Exception as ex:
        return f"❌ {ex}"
