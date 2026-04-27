"""
audio_processor.py
==================
【独立功能模块】用 FFmpeg 裁剪录音中的静音/噪音段。

两种工作模式（由 cfg.audio_processor.re_transcribe_after_cut 控制）：
  A. re_transcribe=True  裁剪后重新跑完整 pipeline（更准，推荐）
  B. re_transcribe=False 仅映射旧时间轴到新时间轴（更快）

独立使用：
  python audio_processor.py D:/audio.mp3           # 裁剪 → 提示重新转写
  python audio_processor.py D:/audio.mp3 --remap   # 裁剪 → 仅映射时间轴
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import cfg
from utils import Segment, derive_output_paths, save_json, save_srt, save_txt


class AudioProcessor:

    def __init__(self) -> None:
        self._cfg     = cfg.audio_processor
        self._out_dir = cfg.paths.processed_audio_dir
        self._check_ffmpeg()

    # ── 公开接口 ─────────────────────────────────────────────

    def cut_silence(self, audio_path: Path | str) -> Path:
        """裁剪静音段，返回裁剪后的音频路径。"""
        audio_path = Path(audio_path)
        out_path   = self._out_dir / f"{audio_path.stem}_cut.{self._cfg.output_format}"

        print(f"\n[AudioProcessor] 静音裁剪: {audio_path.name}")
        silence_ranges = self._detect_silence(audio_path)
        print(f"[AudioProcessor] 检测到 {len(silence_ranges)} 个静音段")

        if not silence_ranges:
            print("[AudioProcessor] 无需裁剪，返回原文件。")
            return audio_path

        keep_ranges = self._invert_silence(silence_ranges, audio_path)
        print(f"[AudioProcessor] 保留 {len(keep_ranges)} 个语音段")
        self._ffmpeg_concat(audio_path, keep_ranges, out_path)
        print(f"[AudioProcessor] 裁剪完成 → {out_path}")
        return out_path

    def remap_segments(
        self,
        original_segments: list[Segment],
        audio_path: Path | str,
    ) -> list[Segment]:
        """模式 B：将原始 segments 的时间轴映射到裁剪后的新时间轴。"""
        silence_ranges = self._detect_silence(Path(audio_path))
        if not silence_ranges:
            return original_segments
        return [
            Segment(
                start=self._remap_time(seg.start, silence_ranges),
                end=self._remap_time(seg.end,   silence_ranges),
                text=seg.text, speaker=seg.speaker, source=seg.source,
                text_whisper=seg.text_whisper, text_whisperx=seg.text_whisperx,
                words=seg.words, extra=seg.extra,
            )
            for seg in original_segments
        ]

    def process_and_save(
        self,
        audio_path: Path | str,
        original_segments: list[Segment] | None = None,
    ) -> tuple[Path, list[Segment] | None]:
        """
        一键：裁剪 → 重新转写或映射时间轴 → 保存输出文件。
        返回 (cut_audio_path, new_segments_or_None)
        """
        cut_path = self.cut_silence(audio_path)

        if self._cfg.re_transcribe_after_cut:
            print("[AudioProcessor] re_transcribe=True，请用 main.py 重新转写裁剪后的音频。")
            return cut_path, None

        if original_segments is None:
            print("[AudioProcessor] ⚠️ re_transcribe=False 但未提供 original_segments，跳过映射。")
            return cut_path, None

        print("[AudioProcessor] 映射时间轴...")
        new_segs = self.remap_segments(original_segments, audio_path)
        out_paths = derive_output_paths(Path(audio_path), cfg.paths.transcripts_dir, "_cut")
        save_srt(new_segs, out_paths["srt"])
        save_txt(new_segs, out_paths["txt"])
        save_json(new_segs, out_paths["json"])
        return cut_path, new_segs

    # ── 私有：FFmpeg 调用 ────────────────────────────────────

    def _detect_silence(self, audio_path: Path) -> list[tuple[float, float]]:
        thresh = self._cfg.silence_thresh_db
        dur    = self._cfg.min_silence_duration_s
        cmd = [
            "ffmpeg", "-i", str(audio_path),
            "-af", f"silencedetect=noise={thresh}dB:d={dur}",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, check=False)
        except FileNotFoundError:
            raise RuntimeError("FFmpeg 未找到，请安装并加入 PATH: https://ffmpeg.org/download.html")

        ranges: list[tuple[float, float]] = []
        start_t: float | None = None
        for line in result.stderr.splitlines():
            if "silence_start" in line:
                try:
                    start_t = float(line.split("silence_start: ")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif "silence_end" in line and start_t is not None:
                try:
                    end_t = float(line.split("silence_end: ")[1].split()[0])
                    ranges.append((start_t, end_t))
                    start_t = None
                except (IndexError, ValueError):
                    pass
        return ranges

    def _invert_silence(self, silence_ranges: list[tuple[float, float]], audio_path: Path) -> list[tuple[float, float]]:
        duration = self._get_duration(audio_path)
        pad = self._cfg.padding_s
        keep: list[tuple[float, float]] = []
        cursor = 0.0
        for s_start, s_end in silence_ranges:
            seg_end = max(0.0, s_start + pad)
            if cursor < seg_end:
                keep.append((cursor, seg_end))
            cursor = max(cursor, s_end - pad)
        if cursor < duration:
            keep.append((cursor, duration))
        return keep

    def _ffmpeg_concat(self, audio_path: Path, keep_ranges: list[tuple[float, float]], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(keep_ranges)
        filter_parts = [
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[s{i}]"
            for i, (start, end) in enumerate(keep_ranges)
        ]
        concat_inputs = "".join(f"[s{i}]" for i in range(n))
        filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[out]")
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]", str(out_path),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

    def _get_duration(self, audio_path: Path) -> float:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)]
        try:
            out  = subprocess.check_output(cmd, text=True)
            data = json.loads(out)
            return float(data["format"]["duration"])
        except Exception:
            return 0.0

    @staticmethod
    def _remap_time(t: float, silence_ranges: list[tuple[float, float]]) -> float:
        removed = 0.0
        for s_start, s_end in silence_ranges:
            if t <= s_start:
                break
            elif t >= s_end:
                removed += s_end - s_start
            else:
                removed += t - s_start
        return max(0.0, t - removed)

    @staticmethod
    def _check_ffmpeg() -> None:
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  [AudioProcessor] FFmpeg 未检测到，静音裁剪功能不可用。")
            print("   安装: https://ffmpeg.org/download.html")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python audio_processor.py <audio_path> [--remap]")
        sys.exit(1)

    import patches
    patches.apply_all()

    audio  = Path(sys.argv[1])
    remap  = "--remap" in sys.argv
    proc   = AudioProcessor()

    if remap:
        json_path = cfg.paths.transcripts_dir / f"{audio.stem}_reconciled.json"
        if not json_path.exists():
            print(f"❌ 未找到对应的转写 JSON: {json_path}")
            sys.exit(1)
        from utils import load_json
        segs = load_json(json_path)
        cut_path, _ = proc.process_and_save(audio, segs)
    else:
        cut_path, _ = proc.process_and_save(audio)

    print(f"\n✅ 完成: {cut_path}")
    if cfg.audio_processor.re_transcribe_after_cut and not remap:
        print(f'💡 重新转写命令: python main.py "{cut_path}"')
