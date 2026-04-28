"""
config.py — Unified configuration. Edit USER SETTINGS before first run.
所有配置集中在这里，首次使用修改 USER SETTINGS 区域。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# ★ USER SETTINGS
# ============================================================

HF_TOKEN: str = "hf_YOUR_TOKEN_HERE"
AUDIO_PATH: str = r"D:\path\to\your\audio.mp3"
DATA_ROOT: str = r"D:\whisper_data"

# ============================================================


@dataclass
class DeviceConfig:
    device: str = "cuda"
    compute_type: str = "float16"  # cuda: "float16"; cpu: "int8" or "float32"


@dataclass
class WhisperConfig:
    model_size: str = "large-v2"
    language: str = "en"
    task: str = "transcribe"  # "transcribe" or "translate"
    verbose: bool = True
    fp16: bool = True


@dataclass
class WhisperXConfig:
    model_size: str = "large-v2"
    language: str = "en"
    batch_size: int = 16       # 4070TiS 12GB: 16; conservative: 8
    beam_size: int = 10
    compute_type: str = "float16"
    return_char_alignments: bool = False
    # VAD — may be overridden by DiarizationPostProcessConfig.scene_preset
    vad_onset: float = 0.3
    vad_offset: float = 0.3
    chunk_size_s: float = 30.0
    # Speaker count hints
    min_speakers: int = 1
    max_speakers: int = 5
    num_speakers: int | None = None  # set if exact count known


@dataclass
class ReconcilerConfig:
    # Tolerance (seconds) for matching segments across models
    time_tolerance_s: float = 1.5
    # Conflict strategy: "keep_both" | "prefer_whisper" | "prefer_whisperx"
    conflict_strategy: str = "keep_both"
    # Hallucination detection: flag segment if same text repeats >= N times
    hallucination_repeat_threshold: int = 3


@dataclass
class SpeakerConfig:
    # Cosine similarity threshold for speaker matching (0~1)
    similarity_threshold: float = 0.80

    # Max auto-learned embeddings per speaker (not a model limit — user-defined)
    # Edge effect diminishes after ~20; set 999999 for unlimited accumulation
    max_embeddings_per_speaker: int = 50

    # Rolling window strategy for auto embeddings: "rolling" | "keep_first"
    embedding_strategy: str = "rolling"

    # Minimum segment duration (s) for reliable embedding extraction
    min_segment_duration_s: float = 0.8

    # Weight of anchor embeddings vs auto embeddings in the reference vector
    # anchor_blend=0.6 means anchors contribute 60% of the final centroid
    anchor_blend: float = 0.6

    # Distance threshold: correction samples closer than this to the centroid
    # are added to auto embeddings; farther ones go to the soft_corrections list
    correction_distance_threshold: float = 0.25

    # Minimum cosine similarity to assign a known speaker during remapping
    # Below this → label as SPEAKER_NEW_XX (unknown speaker)
    remap_min_confidence: float = 0.70


@dataclass
class DiarizationPostProcessConfig:
    enabled: bool = True
    # Scene preset overrides VAD params:
    #   "default"  — use WhisperXConfig values
    #   "dialogue" — dense conversation: lower onset/offset, finer boundaries
    #   "lecture"  — monologue: higher onset/offset, fewer spurious cuts
    scene_preset: str = "default"
    merge_gap_s: float = 0.3       # merge same-speaker segments closer than this
    min_segment_s: float = 0.3     # discard isolated segments shorter than this
    # Overlap resolution: "longer" | "earlier" | "keep"
    overlap_strategy: str = "longer"


@dataclass
class AudioProcessorConfig:
    silence_thresh_db: float = -35.0
    min_silence_duration_s: float = 0.8
    padding_s: float = 0.15
    output_format: str = "mp3"
    re_transcribe_after_cut: bool = True


@dataclass
class PathConfig:
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

    @property
    def pending_mapping_file(self) -> Path:
        """Temporary file written after cold-start diarization, filled by user."""
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root / "pending_mapping.json"


class AppConfig:
    """
    Global config singleton. Import everywhere via: from config import cfg
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
