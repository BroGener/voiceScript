# Whisper Suite

双模型语音转写系统 — Whisper + WhisperX 并行校对，含说话人识别、声纹学习、静音裁剪。

---

## 文件结构

```
whisper_suite/
├── config.py                    # ★ 所有配置集中在这里（路径/Token/参数）
├── patches.py                   # 兼容性补丁（HF/CUDA/PyTorch）
├── utils.py                     # 公共工具（Segment、时间格式、文件输出）
│
├── version_checker.py           # 【独立工具】环境版本检查 & 快照
├── main.py                      # 主入口 — 编排完整 pipeline
│
├── transcriber_whisper.py       # Whisper 转写器
├── transcriber_whisperx.py      # WhisperX 转写器（含对齐 & 说话人分割）
├── diarization_postprocessor.py # 说话人分割后处理（密集对话修复）
├── reconciler.py                # 双模型结果校对合并
├── speaker_manager.py           # 声纹管理（存储 / 加载 / 校正）
│
├── audio_processor.py           # 【独立功能】FFmpeg 静音裁剪
└── correction_tool.py           # 【独立工具】人工校正 → 声纹数据库更新
```

---

## 快速开始

### 第一步：配置

编辑 `config.py`，修改 USER SETTINGS：

```python
HF_TOKEN   = "hf_你的Token"            # HuggingFace Token
AUDIO_PATH = r"D:\path\to\audio.mp3"   # 默认音频路径
DATA_ROOT  = r"D:\whisper_data"         # 数据存储根目录
```

HuggingFace Token 申请：https://huggingface.co/settings/tokens  
需同意许可：`pyannote/speaker-diarization`、`pyannote/segmentation`、`pyannote/embedding`

### 第二步：检查环境（首次强烈推荐）

```bash
# 生成版本快照（当前能跑的环境即为黄金版本）
python version_checker.py

# 他人使用时对比环境
python version_checker.py --check
```

### 第三步：运行转写

```bash
# 双模型 + 校对合并（默认，输出三套结果）
python main.py D:/audio.mp3

# 只跑 Whisper（无说话人标签）
python main.py D:/audio.mp3 --only-whisper

# 只跑 WhisperX（含说话人分割）
python main.py D:/audio.mp3 --only-whisperx
```

---

## 输出文件

保存在 `DATA_ROOT/transcripts/`：

| 文件 | 内容 |
|------|------|
| `<stem>_reconciled.srt/txt/json` | 双模型合并结果（**主要输出**）|
| `<stem>_whisper.srt/txt/json`    | Whisper 单独结果 |
| `<stem>_whisperx.srt/txt/json`   | WhisperX 单独结果（含说话人）|

冲突句格式（两模型识别不一致时）：
```
00:01:23,456 --> 00:01:26,789
  (SPEAKER_00) [WHISPER]  : This is the whisper version
  (SPEAKER_00) [WHISPERX] : This is the whisperx version
```

---

## 说话人识别调优

### 针对密集对话场景（快速交替说话）

在 `config.py` 中修改一行：

```python
cfg.diarization_post.scene_preset = "dialogue"
```

这会自动调低 VAD 敏感度（onset/offset: 0.3 → 0.15），同时后处理器会：
1. 合并同一说话人间隔 < 0.3s 的碎片
2. 移除极短孤立片段（< 0.3s）
3. 解决两人片段时间重叠

其他预设：
- `"lecture"` — 独白/讲座场景，减少呼吸/停顿引起的误切
- `"default"` — 使用 config.py 中的原始参数

### 声纹学习

每次运行后，系统自动提取各说话人声纹存到 `DATA_ROOT/speakers/<name>.json`。  
积累越多，下次识别越稳定。

关于样本上限（`max_samples_per_speaker = 50`）：
- 这是人为上限，不是模型限制
- 边际效益递减：前 20 条贡献最大，之后变化很小
- 想无限积累：设为 `999999` 即可
- 默认 `rolling` 策略：新样本滚动替换最旧的，声纹随时间保鲜

### 人工校正

```bash
# 1. 用文本编辑器打开转写结果，修改括号里的说话人名字
# 2. 运行校正工具
python correction_tool.py D:/audio.mp3
```

校正工具会把正确的声纹加入数据库，把错误的移除，**每次校正都会让下次更准**。

---

## 静音裁剪（可选）

```bash
# 裁剪 → 重新转写（更准，推荐）
python audio_processor.py D:/audio.mp3

# 裁剪 → 仅映射时间轴（更快）
python audio_processor.py D:/audio.mp3 --remap
```

裁剪后的音频在 `DATA_ROOT/processed_audio/`。

---

## 参数速查（config.py）

```python
# WhisperX 性能（4070TiS 推荐）
cfg.whisperx.batch_size  = 16      # 可试 24
cfg.whisperx.beam_size   = 10      # 更大=更准更慢
cfg.whisperx.compute_type = "float16"

# 说话人识别
cfg.diarization_post.scene_preset  = "dialogue"  # 密集对话场景
cfg.diarization_post.merge_gap_s   = 0.3          # 碎片合并间隔阈值
cfg.diarization_post.min_segment_s = 0.3          # 最短有效片段

# 校对合并
cfg.reconciler.time_tolerance_s  = 1.5     # 时间对齐容差（秒）
cfg.reconciler.conflict_strategy = "keep_both"  # 或 "prefer_whisper" / "prefer_whisperx"

# 声纹
cfg.speaker.max_samples_per_speaker = 50    # 样本上限（设 999999 无限积累）
cfg.speaker.similarity_threshold    = 0.80  # 匹配阈值

# 静音裁剪
cfg.audio_processor.silence_thresh_db     = -35.0
cfg.audio_processor.re_transcribe_after_cut = True
```
