"""
version_checker.py
==================
【独立工具】检查并记录当前运行环境的所有关键库版本。

用法：
  python version_checker.py              # 检查并保存快照（首次运行）
  python version_checker.py --check      # 与已保存的快照对比，发现漂移时警告
  python version_checker.py --strict     # 对比模式，不一致时 exit(1)（适合 CI）
  python version_checker.py --snapshot ./my_snapshot.json  # 自定义快照路径

设计：
  - 完全独立，不依赖项目内其他模块（可单独分发）
  - 你当前能运行的版本就是"黄金版本"，首次运行即生成快照
  - 别人首次运行时，自动与快照对比并给出明确提示
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# (import_name, pip_name, 重要程度: "critical" | "warn" | "info")
PACKAGES_TO_CHECK = [
    ("whisper",         "openai-whisper",  "critical"),
    ("whisperx",        "whisperx",        "critical"),
    ("torch",           "torch",           "critical"),
    ("torchaudio",      "torchaudio",      "critical"),
    ("torchvision",     "torchvision",     "warn"),
    ("pyannote.audio",  "pyannote.audio",  "critical"),
    ("transformers",    "transformers",    "critical"),
    ("huggingface_hub", "huggingface-hub", "critical"),
    ("faster_whisper",  "faster-whisper",  "warn"),
    ("numpy",           "numpy",           "warn"),
    ("scipy",           "scipy",           "warn"),
    ("soundfile",       "soundfile",       "warn"),
    ("librosa",         "librosa",         "info"),
    ("ffmpeg",          None,              "warn"),
]


def _get_version(import_name: str) -> str:
    if import_name == "ffmpeg":
        return _get_ffmpeg_version()
    try:
        mod = importlib.import_module(import_name)
        if hasattr(mod, "__version__"):
            return str(mod.__version__)
    except ImportError:
        pass
    try:
        pkg = import_name.replace(".", "-")
        return importlib.metadata.version(pkg)
    except Exception:
        return "NOT_INSTALLED"


def _get_ffmpeg_version() -> str:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-version"], stderr=subprocess.STDOUT, text=True
        )
        parts = out.splitlines()[0].split()
        return parts[2] if len(parts) >= 3 else out[:40]
    except FileNotFoundError:
        return "NOT_INSTALLED"
    except Exception as e:
        return f"ERROR:{e}"


def _get_cuda_info() -> dict:
    info: dict = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["cudnn_version"] = str(torch.backends.cudnn.version())
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 1)
    except ImportError:
        info["cuda_available"] = False
        info["note"] = "torch not installed"
    return info


def collect_versions() -> dict:
    snapshot: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "platform": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python": sys.version,
            "python_short": platform.python_version(),
            "machine": platform.machine(),
        },
        "cuda": _get_cuda_info(),
        "packages": {},
    }

    print("\n" + "=" * 62)
    print("  环境版本检查")
    print("=" * 62)
    print(f"  Python  : {platform.python_version()}")

    cuda = snapshot["cuda"]
    if cuda.get("cuda_available"):
        print(f"  GPU     : {cuda.get('gpu_name')} ({cuda.get('gpu_vram_gb')} GB)")
        print(f"  CUDA    : {cuda.get('cuda_version')}   cuDNN: {cuda.get('cudnn_version')}")
    else:
        print("  GPU     : 不可用（CPU 模式）")

    print("-" * 62)
    print(f"  {'库名':<26} {'版本':<20} {'重要度'}")
    print("-" * 62)

    for import_name, _pip_name, level in PACKAGES_TO_CHECK:
        ver = _get_version(import_name)
        snapshot["packages"][import_name] = {"version": ver, "level": level}
        ok = ver not in ("NOT_INSTALLED",) and not ver.startswith("ERROR")
        icon = "✅" if ok else ("🔴" if level == "critical" else "🟡")
        print(f"  {icon} {import_name:<24} {ver:<20} [{level}]")

    print("=" * 62)
    return snapshot


def save_snapshot(snapshot: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 版本快照已保存至: {path}")


def load_snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_snapshots(current: dict, saved: dict, strict: bool = False) -> bool:
    print("\n" + "=" * 62)
    print("  版本对比（当前 vs 快照）")
    print(f"  快照生成时间: {saved.get('generated_at', '未知')}")
    print("=" * 62)

    cur_pkgs = current.get("packages", {})
    sav_pkgs = saved.get("packages", {})
    mismatches: list[str] = []

    for key in sorted(set(cur_pkgs) | set(sav_pkgs)):
        cur_ver = cur_pkgs.get(key, {}).get("version", "MISSING")
        sav_ver = sav_pkgs.get(key, {}).get("version", "MISSING")
        level   = cur_pkgs.get(key, sav_pkgs.get(key, {})).get("level", "info")

        if cur_ver == sav_ver:
            print(f"  ✅ {key:<26} {cur_ver}")
        else:
            icon = "🔴" if level == "critical" else "🟡"
            print(f"  {icon} {key:<26} 当前: {cur_ver:<15} 快照: {sav_ver}  ← 不一致!")
            mismatches.append(f"{key}: {sav_ver} → {cur_ver}")

    print("=" * 62)

    if mismatches:
        print(f"\n⚠️  发现 {len(mismatches)} 个版本差异:")
        for m in mismatches:
            print(f"     • {m}")
        print(
            "\n💡 如果脚本运行正常，可忽略差异并运行 "
            "`python version_checker.py` 更新快照。\n"
            "   如果出现报错，建议回退到快照中记录的版本。"
        )
        if strict:
            print("\n❌ --strict 模式：版本不一致，退出。")
            sys.exit(1)
        return False

    print("\n🎉 所有版本与快照完全一致，环境正常。")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper Suite 环境版本检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python version_checker.py              # 检查并保存快照
  python version_checker.py --check      # 与快照对比
  python version_checker.py --strict     # 对比，不一致时 exit 1
        """,
    )
    parser.add_argument("--check",    action="store_true", help="与已保存快照对比")
    parser.add_argument("--strict",   action="store_true", help="对比模式，不一致时 exit(1)")
    parser.add_argument("--snapshot", type=str, default=None, help="快照文件路径")
    args = parser.parse_args()

    if args.snapshot:
        snapshot_path = Path(args.snapshot)
    else:
        try:
            from config import cfg
            snapshot_path = cfg.paths.version_file
        except ImportError:
            snapshot_path = Path(__file__).parent / "env_versions.json"

    current = collect_versions()

    if args.check or args.strict:
        saved = load_snapshot(snapshot_path)
        if saved is None:
            print(f"\n⚠️  未找到快照文件: {snapshot_path}")
            print("   请先运行 `python version_checker.py` 生成快照。")
            if args.strict:
                sys.exit(1)
        else:
            compare_snapshots(current, saved, strict=args.strict)
    else:
        save_snapshot(current, snapshot_path)


if __name__ == "__main__":
    main()
