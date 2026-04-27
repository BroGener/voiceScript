"""
config.py
=========
统一配置文件 — 所有路径、Token、模型参数都集中在这里。
首次使用请修改 USER SETTINGS 区域，其余部分无需改动。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# ★ USER SETTINGS — 修改这里 ★
# ============================================================

# HuggingFace Token（用于说话人分割模型下载）
# 申请地址：https://huggingface.co/settings/tokens
# 需同意以下模型许可：pyannote/speaker-diarization, pyannote/segmentation, pyannote/embedding
HF_TOKEN: str = "hf_YOUR_TOKEN_HERE"

# 默认音频文件路径（也可在命令行 / 调用时传入覆盖）
AUDIO_PATH: str = r"D:\path\to\your\audio.mp3"

# 本地数据存储根目录（声纹、转写结果、版本信息等均存于此）
DATA_ROOT: str = r"D:\whisper_data"

# ============================================================


@dataclass
class DeviceConfig:
    device: str = "cuda"
    # cuda 推荐 "float16"（省显存约 1/3），显存吃紧用 "int8_float16"
    # cpu 只能用 "int8" 或 "float32"
    compute_type: str = "float16"


@dataclass
class WhisperConfig:
    """原版 openai-whisper 参数"""
    model_size: str = "large-v2"
    language: str = "en"
    task: str = "transcribe"   # "transcribe" 保留原语言 / "translate" 翻译成英文
    verbose: bool = True
    fp16: bool = True          # GPU 下开启加速，CPU 时会自动降回 False


@dataclass
class WhisperXConfig:
    """WhisperX 参数"""
    model_size: str = "large-v2"
    language: str = "en"
    # 4070TiS 12GB 推荐 batch_size=16；保守或显存吃紧用 8
    batch_size: int = 16
    beam_size: int = 10        # 越大越准越慢，4070TiS 推荐 10
    compute_type: str = "float16"
    return_char_alignments: bool = False
    # VAD（语音活动检测）— 会被 DiarizationPostProcessConfig.scene_preset 覆盖
    vad_onset: float = 0.3
    vad_offset: float = 0.3
    chunk_size_s: float = 30.0
    # 说话人分割
    min_speakers: int = 1
    max_speakers: int = 5
    # 如果已知精确人数可设置，覆盖 min/max（None = 自动）
    num_speakers: int | None = None


@dataclass
class ReconcilerConfig:
    """双模型结果校对合并参数"""
    # 时间轴匹配容差（秒）：两模型 segment 时间差在此范围内视为同一句话
    time_tolerance_s: float = 1.5
    # 冲突时的处理策略：
    #   "keep_both"       冲突句保留两个版本，标注来源（推荐）
    #   "prefer_whisper"  英语原声优先用 Whisper
    #   "prefer_whisperx" 需要说话人标签时优先用 WhisperX
    conflict_strategy: str = "keep_both"


@dataclass
class SpeakerConfig:
    """声纹管理参数"""
    # 声纹余弦相似度阈值（0~1），越高越严格；建议 0.75~0.85
    similarity_threshold: float = 0.80

    # 每个说话人最多保留的声纹样本数
    # ─────────────────────────────────────────────────────────
    # 这不是模型限制，是人为上限。理论上可以无限积累。
    # 边际效益递减：前 20 条贡献最大，之后改变越来越小。
    # 上限防止早期错误样本被永久固化。
    # 如果确认样本质量高，可以调到 200+，甚至设 999999 无限积累。
    # ─────────────────────────────────────────────────────────
    max_samples_per_speaker: int = 50

    # 样本保留策略：
    #   "rolling"    滚动窗口，新样本挤掉最旧的（推荐）
    #   "keep_first" 保留最早的样本（声纹非常稳定时使用）
    sample_strategy: str = "rolling"

    # 最短有效声纹片段（秒）：短于此的片段提取的嵌入噪声大，跳过
    min_segment_duration_s: float = 0.8


@dataclass
class DiarizationPostProcessConfig:
    """
    说话人分割后处理参数。
    专门针对"对话密集/快速交替"场景的补救措施。

    根本原因：pyannote VAD 用 onset/offset 双阈值判断发言边界，
    两人快速交替时边界检测不准，导致多人话语被归给同一人。
    后处理在 pyannote 原始输出之后插入规则修复，不需要修改模型。
    """

    # ── 是否启用 ──────────────────────────────────────────────
    enabled: bool = True

    # ── 场景预设（会覆盖 WhisperXConfig 的 VAD 参数）─────────
    #   "default"   使用 WhisperXConfig 中的参数，不覆盖
    #   "dialogue"  密集对话：降低 onset/offset，更细腻的边界切割
    #   "lecture"   独白/讲座：提高 onset/offset，减少呼吸/停顿引起的误切
    scene_preset: str = "default"

    # ── 短片段合并 ────────────────────────────────────────────
    # 同一说话人的相邻片段，间隔小于此值时合并（秒）
    # 解决：快速对话时同一人的话被切成多个短碎片
    merge_gap_s: float = 0.3

    # 时长小于此值的孤立片段视为噪声，并入最近邻居（秒）
    min_segment_s: float = 0.3

    # ── 重叠语音处理 ──────────────────────────────────────────
    # 两个说话人片段时间重叠时，重叠部分的处理策略：
    #   "longer"   归给时长更长的那个说话人（默认）
    #   "earlier"  归给开始时间更早的说话人
    #   "keep"     保留重叠，不处理（原始行为）
    overlap_strategy: str = "longer"


@dataclass
class AudioProcessorConfig:
    """FFmpeg 静音裁剪参数"""
    # 静音检测阈值（dB），越负越安静，-30 ~ -50 合理
    silence_thresh_db: float = -35.0
    # 最短静音片段（秒），短于此不裁剪
    min_silence_duration_s: float = 0.8
    # 裁剪后在每段首尾保留的 padding（秒），避免切掉起始音素
    padding_s: float = 0.15
    output_format: str = "mp3"
    # True = 裁剪后重新转写（更准）/ False = 仅映射时间轴（更快）
    re_transcribe_after_cut: bool = True


@dataclass
class PathConfig:
    """路径派生（自动创建目录，无需手动 mkdir）"""
    _root: Path = field(default_factory=lambda: Path(DATA_ROOT))

    def _ensure(self, sub: str) -> Path:
        p = self._root / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_root(self) -> Path:
        return self._root

    @property
    def speakers_dir(self) -> Path:
        return self._ensure("speakers")

    @property
    def transcripts_dir(self) -> Path:
        return self._ensure("transcripts")

    @property
    def processed_audio_dir(self) -> Path:
        return self._ensure("processed_audio")

    @property
    def version_file(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root / "env_versions.json"


class AppConfig:
    """
    全局配置单例，整个项目统一 import。

    用法：
        from config import cfg
        print(cfg.whisperx.batch_size)
        print(cfg.paths.speakers_dir)
        cfg.diarization_post.scene_preset = "dialogue"  # 运行时覆盖
    """

    def __init__(self) -> None:
        self.hf_token: str = HF_TOKEN
        self.audio_path: Path = Path(AUDIO_PATH)
        self.device: DeviceConfig = DeviceConfig()
        self.whisper: WhisperConfig = WhisperConfig()
        self.whisperx: WhisperXConfig = WhisperXConfig()
        self.reconciler: ReconcilerConfig = ReconcilerConfig()
        self.speaker: SpeakerConfig = SpeakerConfig()
        self.diarization_post: DiarizationPostProcessConfig = DiarizationPostProcessConfig()
        self.audio_processor: AudioProcessorConfig = AudioProcessorConfig()
        self.paths: PathConfig = PathConfig()


cfg = AppConfig()
