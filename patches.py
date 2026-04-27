"""
patches.py
==========
所有运行时兼容性补丁，集中管理。
在任何其他库导入之前 import 此模块即可。

包含：
  1. HuggingFace use_auth_token → token 参数名修复
  2. NVIDIA DLL 路径注入（解决 Windows 上 cublas64_12.dll 找不到）
  3. PyTorch 2.6 安全检查绕过（weights_only 问题）
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
    try:
        import torch
        _orig_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _patched_load
        if hasattr(torch, "serialization"):
            torch.serialization.load = _patched_load
    except ImportError:
        pass


def apply_all() -> None:
    """在程序入口处调用一次即可。"""
    _apply_nvidia_dll_patch()  # DLL 路径要在 torch 导入前完成
    _apply_torch_patch()
    _apply_hf_patch()
