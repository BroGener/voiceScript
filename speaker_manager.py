"""
speaker_manager.py
==================
说话人声纹管理器。

功能：
  1. 提取并持久化声纹嵌入到本地 JSON（跨文件使用）
  2. 加载已知说话人，供下次转写时参考
  3. 人工校正：发现归类错误时更新数据库，使识别越来越准
  4. 余弦相似度匹配

关于样本积累：
  max_samples_per_speaker 是人为上限，不是模型限制。
  声纹嵌入维度约 192~512 维，一条只有几 KB，积累几百条也只占几 MB。
  边际效益递减规律：前 20 条贡献最大，之后改变越来越小。
  上限的意义是防止早期错误样本固化，用 rolling 策略可以让声纹
  随时间"保鲜"。如需无限积累，把 max_samples_per_speaker 设为 999999。

数据格式（DATA_ROOT/speakers/<name>.json）：
  {
    "name": "Alice",
    "embeddings": [[...], ...],
    "corrections": 0,
    "created_at": "...",
    "updated_at": "..."
  }
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import cfg
from utils import Segment


class SpeakerManager:

    def __init__(self) -> None:
        self._dir   = cfg.paths.speakers_dir
        self._scfg  = cfg.speaker
        self._db: dict[str, dict] = {}
        self._load_all()

    # ── 公开：声纹提取 & 存储 ────────────────────────────────

    def extract_and_save(
        self,
        audio_path: Path | str,
        diarize_records: list[dict],
        name_map: dict[str, str] | None = None,
    ) -> None:
        """
        从音频中提取各说话人的声纹嵌入并保存。

        Parameters
        ----------
        audio_path      : 原始音频路径
        diarize_records : [{speaker, start, end}, ...] 来自 WhisperXTranscriber
        name_map        : {"SPEAKER_00": "Alice", ...} 可选，未提供则用自动编号
        """
        audio_path = Path(audio_path)
        name_map   = name_map or {}
        print(f"\n[SpeakerManager] 提取声纹嵌入 ({len(diarize_records)} 段)...")

        embeddings_by_speaker: dict[str, list[np.ndarray]] = {}

        try:
            from pyannote.audio import Inference
            from pyannote.audio import Model as PyanModel
            import torchaudio

            embed_model = PyanModel.from_pretrained(
                "pyannote/embedding", use_auth_token=cfg.hf_token
            )
            inference  = Inference(embed_model, window="whole")
            waveform, sr = torchaudio.load(str(audio_path))

            for rec in diarize_records:
                sp_id   = rec["speaker"]
                start_s = float(rec["start"])
                end_s   = float(rec["end"])

                if end_s - start_s < self._scfg.min_segment_duration_s:
                    continue  # 太短，嵌入噪声大

                s = int(start_s * sr)
                e = int(end_s   * sr)
                chunk = waveform[:, s:e]
                emb = inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr})
                embeddings_by_speaker.setdefault(sp_id, []).append(np.array(emb))

        except Exception as ex:
            print(f"[SpeakerManager] ⚠️ 声纹提取失败（跳过）: {ex}")
            return

        for sp_id, embs in embeddings_by_speaker.items():
            real_name = name_map.get(sp_id, sp_id)
            self._add_embeddings(real_name, embs)

        self._save_all()
        print(f"[SpeakerManager] 声纹已保存: {list(embeddings_by_speaker.keys())}")

    # ── 公开：加载已知说话人 ─────────────────────────────────

    def known_speakers(self) -> dict[str, np.ndarray]:
        """返回 {name: 平均嵌入向量}。"""
        result = {}
        for name, data in self._db.items():
            embs = data.get("embeddings", [])
            if embs:
                result[name] = np.mean([np.array(e) for e in embs], axis=0)
        return result

    def list_speakers(self) -> list[str]:
        return list(self._db.keys())

    # ── 公开：人工校正 ───────────────────────────────────────

    def correct_segment(
        self,
        audio_path: Path | str,
        segment: Segment,
        correct_speaker: str,
    ) -> None:
        """
        人工校正单条 segment 的说话人归类。

        流程：
          1. 从音频中提取该 segment 的声纹嵌入
          2. 将嵌入加入 correct_speaker 的数据库
          3. 从 segment.speaker（原错误说话人）数据库中移除最相似的嵌入
        """
        audio_path = Path(audio_path)
        print(f"[SpeakerManager] 校正: {segment.speaker} → {correct_speaker} "
              f"({segment.start:.1f}s - {segment.end:.1f}s)")

        emb = self._extract_single(audio_path, segment.start, segment.end)
        if emb is None:
            print("[SpeakerManager] ⚠️ 无法提取声纹，校正跳过。")
            return

        self._add_embeddings(correct_speaker, [emb])

        wrong = segment.speaker
        if wrong in self._db and wrong != correct_speaker:
            self._remove_closest(wrong, emb)
            self._db[wrong]["corrections"] = self._db[wrong].get("corrections", 0) + 1

        if correct_speaker in self._db:
            self._db[correct_speaker]["corrections"] = (
                self._db[correct_speaker].get("corrections", 0) + 1
            )

        self._save_all()
        print(f"[SpeakerManager] 校正完成，{correct_speaker} 声纹已更新。")

    def apply_corrections_from_file(
        self,
        audio_path: Path | str,
        corrected_segs: list[Segment],
        original_segs: list[Segment],
    ) -> None:
        """批量校正：对比 original 和 corrected，找出说话人被改动的条目，逐条更新。"""
        changed = 0
        for orig, corr in zip(original_segs, corrected_segs):
            if orig.speaker != corr.speaker and corr.speaker:
                self.correct_segment(audio_path, orig, corr.speaker)
                changed += 1
        print(f"[SpeakerManager] 批量校正完成，共 {changed} 条更新。")

    # ── 公开：相似度匹配 ─────────────────────────────────────

    def match_speaker(self, embedding: np.ndarray) -> str | None:
        best_name: str | None = None
        best_score = -1.0
        for name, data in self._db.items():
            embs = data.get("embeddings", [])
            if not embs:
                continue
            avg   = np.mean([np.array(e) for e in embs], axis=0)
            score = float(self._cos(embedding, avg))
            if score > best_score:
                best_score = score
                best_name  = name
        if best_score >= self._scfg.similarity_threshold:
            return best_name
        return None

    # ── 私有：数据库 I/O ─────────────────────────────────────

    def _load_all(self) -> None:
        self._db = {}
        for fp in self._dir.glob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._db[data["name"]] = data
            except Exception as ex:
                print(f"[SpeakerManager] 加载 {fp.name} 失败: {ex}")
        if self._db:
            print(f"[SpeakerManager] 已加载 {len(self._db)} 位说话人: {list(self._db.keys())}")

    def _save_all(self) -> None:
        for name, data in self._db.items():
            data["updated_at"] = datetime.now().isoformat(timespec="seconds")
            safe = name.replace(" ", "_").replace("/", "_")
            fp = self._dir / f"{safe}.json"
            fp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _add_embeddings(self, name: str, embs: list[np.ndarray]) -> None:
        if name not in self._db:
            self._db[name] = {
                "name": name,
                "embeddings": [],
                "corrections": 0,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

        existing: list = self._db[name]["embeddings"]
        for emb in embs:
            existing.append(emb.tolist())

        max_n    = self._scfg.max_samples_per_speaker
        strategy = self._scfg.sample_strategy

        # ── 样本保留策略 ──────────────────────────────────────
        # max_samples_per_speaker 是人为上限，不是模型限制。
        # 边际效益递减：前 20 条贡献最大，之后改变越来越小。
        # "rolling"（默认）: 新样本挤掉最旧的，声纹随时间保鲜。
        # "keep_first"     : 保留最早的，适合声纹非常稳定的情况。
        # 想无限积累: 把 max_samples_per_speaker 设为 999999。
        # ─────────────────────────────────────────────────────
        if len(existing) > max_n:
            self._db[name]["embeddings"] = (
                existing[:max_n] if strategy == "keep_first" else existing[-max_n:]
            )

    def _remove_closest(self, name: str, target: np.ndarray) -> None:
        embs = self._db[name].get("embeddings", [])
        if not embs:
            return
        scores = [self._cos(np.array(e), target) for e in embs]
        embs.pop(int(np.argmax(scores)))

    def _extract_single(self, audio_path: Path, start_s: float, end_s: float) -> np.ndarray | None:
        try:
            from pyannote.audio import Inference
            from pyannote.audio import Model as PyanModel
            import torchaudio

            embed_model = PyanModel.from_pretrained(
                "pyannote/embedding", use_auth_token=cfg.hf_token
            )
            inference = Inference(embed_model, window="whole")
            waveform, sr = torchaudio.load(str(audio_path))
            chunk = waveform[:, int(start_s * sr):int(end_s * sr)]
            return np.array(inference({"waveform": chunk.unsqueeze(0), "sample_rate": sr}))
        except Exception as ex:
            print(f"[SpeakerManager] 提取嵌入失败: {ex}")
            return None

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0
