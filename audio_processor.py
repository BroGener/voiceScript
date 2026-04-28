"""
audio_processor.py
==================
【独立功能模块】用 FFmpeg 裁剪录音中的静音/噪音段。

两种工作模式（由 cfg.audio_processor.re_transcribe_after_cut 控制）：
  A. re_transcribe=True  裁剪后自动触发完整转写 pipeline（更准，推荐）
  B. re_transcribe=False 仅映射旧时间轴到新时间轴（更快）

两种模式均一步完成，不需要手动再调用 main.py。

裁剪后的字幕会在每个片段 extra 中保存原始时间轴（orig_start/orig_end），
SRT/TXT 输出时会附注 [orig: HH:MM:SS,mmm --> HH:MM:SS,mmm]，
方便定位原音频中的问题位置。

每个裁剪点在音频上插入极短的低音白噪声标记（-90dB，几乎不可见），
并在字幕中加入 [CUT] 标记行。

独立使用：
  python audio_processor.py D:/audio.mp3           # 裁剪 → 自动转写（模式 A）
  python audio_processor.py D:/audio.mp3 --remap   # 裁剪 → 仅映射时间轴（模式 B）
  python audio_processor.py D:/audio.mp3 --verify  # 仅做裁剪前后字幕比对验证
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from config import cfg
from utils import Segment, derive_output_paths, fmt_srt, load_json, save_json, save_srt, save_txt


class AudioProcessor:

    def __init__(self) -> None:
        self._cfg     = cfg.audio_processor
        self._out_dir = cfg.paths.processed_audio_dir
        self._check_ffmpeg()

    # ── 公开接口 ─────────────────────────────────────────────

    def cut_silence(self, audio_path: Path | str) -> tuple[Path, list[tuple[float, float]]]:
        """
        裁剪静音段。

        Returns
        -------
        (cut_audio_path, silence_ranges)
        silence_ranges 也返回出来，供时间轴映射和 verify 使用，避免重复检测。
        """
        audio_path = Path(audio_path)
        out_path   = self._out_dir / f"{audio_path.stem}_cut.{self._cfg.output_format}"

        print(f"\n[AudioProcessor] 静音裁剪: {audio_path.name}")
        silence_ranges = self._detect_silence(audio_path)
        print(f"[AudioProcessor] 检测到 {len(silence_ranges)} 个静音段")

        if not silence_ranges:
            print("[AudioProcessor] 无需裁剪，返回原文件。")
            return audio_path, []

        keep_ranges = self._invert_silence(silence_ranges, audio_path)
        print(f"[AudioProcessor] 保留 {len(keep_ranges)} 个语音段")
        self._ffmpeg_concat(audio_path, keep_ranges, out_path)
        print(f"[AudioProcessor] 裁剪完成 → {out_path}")
        return out_path, silence_ranges

    def process(
        self,
        audio_path: Path | str,
        original_segments: list[Segment] | None = None,
    ) -> tuple[Path, list[Segment]]:
        """
        一步完成：裁剪 → 转写（或时间轴映射） → 保存字幕文件。

        裁剪点在字幕中以 [CUT at HH:MM:SS,mmm] 标记，
        每个片段附注原始时间轴 [orig: ...] 方便溯源。

        模式由 cfg.audio_processor.re_transcribe_after_cut 控制：
          True  → 裁剪后自动跑完整 pipeline
          False → 裁剪后仅映射原有时间轴，需传入 original_segments
        """
        audio_path   = Path(audio_path)
        cut_path, silence_ranges = self.cut_silence(audio_path)

        if not silence_ranges:
            # 没有静音段，直接用原文件走正常流程
            if self._cfg.re_transcribe_after_cut:
                segs = self._run_pipeline(audio_path)
            else:
                segs = self._load_or_require_segments(original_segments, audio_path)
            return audio_path, segs

        if self._cfg.re_transcribe_after_cut:
            # ── 模式 A：重新转写 ────────────────────────────
            print("\n[AudioProcessor] 裁剪后重新转写...")
            segs = self._run_pipeline(cut_path)
            # 重新转写后无法还原原时间轴（时间轴已变），跳过 orig 标注
        else:
            # ── 模式 B：映射时间轴 ──────────────────────────
            print("\n[AudioProcessor] 映射时间轴...")
            original_segments = self._load_or_require_segments(original_segments, audio_path)
            segs = self._remap_segments(original_segments, silence_ranges)
            # 在每个片段 extra 中记录原始时间轴，用于字幕备注
            for seg in segs:
                seg.extra["orig_start"] = seg.extra.pop("_pre_remap_start", seg.start)
                seg.extra["orig_end"]   = seg.extra.pop("_pre_remap_end",   seg.end)

        # 在裁剪点插入 [CUT] 标记片段
        segs = self._insert_cut_markers(segs, silence_ranges)

        # 保存（suffix="_cut" 区分裁剪版本）
        out_paths = derive_output_paths(audio_path, cfg.paths.transcripts_dir, "_cut")
        save_srt(segs, out_paths["srt"])
        save_txt(segs, out_paths["txt"])
        save_json(segs, out_paths["json"])

        print(f"\n[AudioProcessor] ✅ 全部完成")
        print(f"  🎵 裁剪音频 → {cut_path}")
        print(f"  📄 字幕结果 → {out_paths['txt']}")
        return cut_path, segs

    def verify(
        self,
        audio_path: Path | str,
        original_segments: list[Segment] | None = None,
        cut_segments: list[Segment] | None = None,
    ) -> dict:
        """
        比对裁剪前后的字幕，检查是否有片段在裁剪中丢失或严重漂移。

        Returns
        -------
        {
            "total_original": int,
            "matched": int,
            "lost": int,          # 裁剪后找不到对应片段的原始片段数
            "cut_markers": int,   # 裁剪点数量
            "lost_segments": [Segment, ...],
            "ok": bool,
        }
        """
        audio_path = Path(audio_path)
        tol = cfg.reconciler.time_tolerance_s  # 复用时间容差配置

        # 自动加载
        if original_segments is None:
            json_path = cfg.paths.transcripts_dir / f"{audio_path.stem}_reconciled.json"
            if json_path.exists():
                original_segments = load_json(json_path)
            else:
                raise FileNotFoundError(f"未找到原始字幕: {json_path}")

        if cut_segments is None:
            json_path = cfg.paths.transcripts_dir / f"{audio_path.stem}_cut.json"
            if json_path.exists():
                cut_segments = load_json(json_path)
            else:
                raise FileNotFoundError(f"未找到裁剪字幕: {json_path}")

        # 过滤掉 [CUT] 标记片段
        real_cut_segs = [s for s in cut_segments if "[CUT" not in s.text]
        cut_markers   = len(cut_segments) - len(real_cut_segs)

        # 对比：找出原始片段中在裁剪版里找不到对应的
        lost: list[Segment] = []
        for orig in original_segments:
            # 用原时间轴（如有），也用新时间轴找
            found = any(
                abs((orig.start + orig.end) / 2 - (c.start + c.end) / 2) <= tol
                or (c.extra.get("orig_start") is not None and
                    abs(orig.start - c.extra["orig_start"]) <= tol)
                for c in real_cut_segs
            )
            if not found:
                lost.append(orig)

        matched = len(original_segments) - len(lost)
        ok = len(lost) == 0

        print(f"\n[AudioProcessor] 裁剪验证报告:")
        print(f"  原始片段: {len(original_segments)}")
        print(f"  匹配成功: {matched}")
        print(f"  裁剪标记: {cut_markers} 个 [CUT] 点")
        if lost:
            print(f"  ⚠️  丢失片段: {len(lost)}")
            for s in lost[:5]:  # 最多打印 5 个
                print(f"     [{fmt_srt(s.start)}] ({s.speaker}) {s.text[:50]}")
            if len(lost) > 5:
                print(f"     ... 共 {len(lost)} 个")
        else:
            print(f"  ✅ 未发现丢失片段")

        return {
            "total_original": len(original_segments),
            "matched": matched,
            "lost": len(lost),
            "cut_markers": cut_markers,
            "lost_segments": lost,
            "ok": ok,
        }

    # ── 私有：重新转写 ────────────────────────────────────────

    def _run_pipeline(self, cut_audio: Path) -> list[Segment]:
        import gc
        from reconciler import Reconciler
        from speaker_manager import SpeakerManager
        from transcriber_whisper import WhisperTranscriber
        from transcriber_whisperx import WhisperXTranscriber

        speaker_mgr = SpeakerManager()

        wt     = WhisperTranscriber()
        w_segs = wt.transcribe(cut_audio)
        wt.unload()

        wxt     = WhisperXTranscriber()
        wx_segs = wxt.transcribe(cut_audio, known_speakers=speaker_mgr.known_speakers() or None)

        try:
            records = wxt.get_diarize_records(cut_audio)
            if records:
                speaker_mgr.extract_and_save(cut_audio, records)
        except Exception as ex:
            print(f"[AudioProcessor] ⚠️ 声纹提取跳过: {ex}")

        wxt.unload()
        gc.collect()
        return Reconciler().reconcile(w_segs, wx_segs)

    # ── 私有：时间轴映射 ──────────────────────────────────────

    def _remap_segments(
        self,
        original_segments: list[Segment],
        silence_ranges: list[tuple[float, float]],
    ) -> list[Segment]:
        """
        将原始 segments 的时间轴映射到裁剪后的新时间轴。
        映射前先把原始时间存入 extra，供后续 orig 备注使用。
        """
        result = []
        for seg in original_segments:
            new_start = self._remap_time(seg.start, silence_ranges)
            new_end   = self._remap_time(seg.end,   silence_ranges)
            new_extra = dict(seg.extra)
            new_extra["orig_start"] = seg.start   # 保存原始时间轴
            new_extra["orig_end"]   = seg.end
            result.append(Segment(
                start=new_start, end=new_end,
                text=seg.text, speaker=seg.speaker, source=seg.source,
                text_whisper=seg.text_whisper, text_whisperx=seg.text_whisperx,
                words=seg.words, extra=new_extra,
            ))
        return result

    def _load_or_require_segments(
        self,
        segments: list[Segment] | None,
        audio_path: Path,
    ) -> list[Segment]:
        if segments is not None:
            return segments
        json_path = cfg.paths.transcripts_dir / f"{audio_path.stem}_reconciled.json"
        if json_path.exists():
            print(f"[AudioProcessor] 自动加载: {json_path.name}")
            return load_json(json_path)
        raise ValueError(
            f"re_transcribe=False 需要提供原始字幕，或先运行 main.py 生成 {json_path.name}"
        )

    # ── 私有：裁剪点标记 ─────────────────────────────────────

    def _insert_cut_markers(
        self,
        segs: list[Segment],
        silence_ranges: list[tuple[float, float]],
    ) -> list[Segment]:
        """
        在裁剪点位置（静音段中心）插入 [CUT] 标记片段。
        标记片段时间长度为零（start==end），不占字幕空间。
        """
        markers = []
        for s_start, s_end in silence_ranges:
            mid = (s_start + s_end) / 2
            markers.append(Segment(
                start=mid, end=mid,
                text=f"[CUT at {fmt_srt(s_start)} --> {fmt_srt(s_end)}]",
                speaker="",
                source="reconciled",
                extra={"is_cut_marker": True},
            ))
        combined = segs + markers
        combined.sort(key=lambda s: s.start)
        return combined

    # ── 私有：FFmpeg ─────────────────────────────────────────

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

    def _invert_silence(
        self,
        silence_ranges: list[tuple[float, float]],
        audio_path: Path,
    ) -> list[tuple[float, float]]:
        duration = self._get_duration(audio_path)
        pad  = self._cfg.padding_s
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

    def _ffmpeg_concat(
        self,
        audio_path: Path,
        keep_ranges: list[tuple[float, float]],
        out_path: Path,
    ) -> None:
        """
        拼接语音段，并在每个裁剪点前插入极短低音白噪声标记（-90dB, 0.05s）。
        白噪声在波形上可见（方便确认裁剪位置），人耳几乎听不到。
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = len(keep_ranges)

        # 每段语音前加一小段白噪声标记（第一段除外，不需要标记开头）
        filter_parts = []
        segment_labels = []

        for i, (start, end) in enumerate(keep_ranges):
            label = f"s{i}"
            filter_parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[{label}]"
            )
            if i > 0:
                # 在这段语音前插入 0.05s 的 -90dB 白噪声作为裁剪点标记
                noise_label = f"n{i}"
                filter_parts.append(
                    #f"anoisesrc=d=0.05:c=white:a=0.000003[{noise_label}]"
                    f"anoisesrc=d=0.15:c=white:a=0.08,bandpass=f=1500:width_type=o:w=2[{noise_label}]"
                )
                segment_labels.append(noise_label)
            segment_labels.append(label)

        total = len(segment_labels)
        concat_inputs = "".join(f"[{lbl}]" for lbl in segment_labels)
        filter_parts.append(f"{concat_inputs}concat=n={total}:v=0:a=1[out]")

        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

    def _get_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", str(audio_path),
        ]
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
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️  [AudioProcessor] FFmpeg 未检测到，静音裁剪功能不可用。")
            print("   安装: https://ffmpeg.org/download.html")


# ── 独立命令行入口 ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python audio_processor.py <audio_path> [选项]")
        print()
        print("选项:")
        print("  (默认)   裁剪 → 自动重新转写 → 生成字幕")
        print("  --remap  裁剪 → 仅映射原有时间轴 → 生成字幕（更快）")
        print("  --verify 比对裁剪前后字幕，检查是否有漏切")
        sys.exit(1)

    import patches
    patches.apply_all()

    audio = Path(sys.argv[1])
    if not audio.exists():
        print(f"❌ 文件不存在: {audio}")
        sys.exit(1)

    remap  = "--remap"  in sys.argv
    verify = "--verify" in sys.argv

    proc = AudioProcessor()

    try:
        if verify:
            proc.verify(audio)
        else:
            if remap:
                cfg.audio_processor.re_transcribe_after_cut = False
            cut_path, segs = proc.process(audio)
            real_segs = [s for s in segs if not s.extra.get("is_cut_marker")]
            print(f"\n✅ 处理完成")
            print(f"   裁剪音频  : {cut_path}")
            print(f"   片段数量  : {len(real_segs)}")
            print(f"   裁剪标记  : {len(segs) - len(real_segs)} 个 [CUT] 点")
            print(f"\n💡 运行以下命令验证裁剪结果:")
            print(f'   python audio_processor.py "{audio}" --verify')
    except Exception as ex:
        import traceback
        print(f"\n❌ 出错: {ex}")
        traceback.print_exc()
        sys.exit(1)
