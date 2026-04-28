"""
patches.py
==========
Runtime compatibility patches. Import before anything else.

Patches:
  1. HuggingFace use_auth_token → token rename
  2. NVIDIA DLL path injection (Windows: fixes cublas64_12.dll not found)
  3. PyTorch 2.6+ weights_only bypass for pyannote/lightning model loading
"""

from __future__ import annotations
import os
import site
import sys


def _apply_hf_patch() -> None:
    try:
        import huggingface_hub
        _orig = huggingface_hub.hf_hub_download

        def _patched(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return _orig(*args, **kwargs)

        huggingface_hub.hf_hub_download = _patched
    except ImportError:
        pass


def _apply_nvidia_dll_patch() -> None:
    if sys.platform != "win32":
        return
    for path in site.getsitepackages():
        nvidia_root = os.path.join(path, "nvidia")
        if not os.path.exists(nvidia_root):
            continue
        for root, dirs, _ in os.walk(nvidia_root):
            if "bin" in dirs:
                bin_path = os.path.abspath(os.path.join(root, "bin"))
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                try:
                    os.add_dll_directory(bin_path)
                except (AttributeError, OSError):
                    pass


def _apply_torch_patch() -> None:
    """
    Fix PyTorch 2.6+ weights_only=True breaking pyannote/lightning model loading.

    Root cause: lightning_fabric._load() calls torch.load(..., weights_only=True)
    explicitly. Our patch must FORCE weights_only=False, not just setdefault.

    Two-layer approach:
      1. Force weights_only=False in the patched torch.load wrapper.
      2. Whitelist omegaconf types via add_safe_globals as a fallback,
         in case any caller internally uses weights_only=True through
         a reference we cannot intercept.
    """
    try:
        import torch

        # Layer 1: whitelist omegaconf globals that pyannote checkpoints contain
        try:
            import omegaconf
            from omegaconf import DictConfig, ListConfig
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([DictConfig, ListConfig])
        except (ImportError, Exception):
            pass

        # Layer 2: force weights_only=False on every torch.load call
        # Use FORCE (not setdefault) because lightning passes weights_only=True explicitly
        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False   # force, not setdefault
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load

        # Also patch the internal serialization module reference
        if hasattr(torch, "serialization"):
            torch.serialization.load = _patched_load

        # Patch lightning_fabric directly if already imported
        # (handles cases where lightning cached the reference before our patch)
        try:
            import lightning_fabric.utilities.cloud_io as _lf_io
            _lf_io._load = _patched_load
        except (ImportError, AttributeError):
            pass

        try:
            import pytorch_lightning.utilities.cloud_io as _pl_io
            _pl_io._load = _patched_load
        except (ImportError, AttributeError):
            pass

    except ImportError:
        pass


def apply_all() -> None:
    """Call once at program entry, before any other imports."""
    _apply_nvidia_dll_patch()   # must run before torch import
    _apply_torch_patch()
    _apply_hf_patch()
