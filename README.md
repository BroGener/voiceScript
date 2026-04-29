# 🎙️ Speech Processing & Speaker Diarization System

A modular Python pipeline for **speech transcription, speaker diarization, and voiceprint learning**, built on top of Whisper and WhisperX.

This project focuses on **accuracy, post-processing, and iterative improvement**, rather than raw transcription.

---

## 🚀 Features

### 🔹 Dual-Model Transcription
- Integrates **OpenAI Whisper** (accurate timestamps)
- Uses **WhisperX** for alignment and speaker diarization
- Custom **Reconciler** merges outputs intelligently

### 🔹 Speaker Recognition & Learning
- Speaker embedding with cosine similarity matching
- Persistent voiceprint database
- Supports iterative learning from corrections

### 🔹 Audio Preprocessing
- Silence detection and removal (FFmpeg)
- Optional:
  - Re-transcription (higher accuracy)
  - Timeline remapping (faster)

### 🔹 Manual Correction Workflow
- Edit `.srt` files to fix speaker labels
- Apply corrections to update voiceprint database
- Confidence-based embedding updates

### 🔹 Post-processing Pipeline
- Overlap resolution
- Short segment removal
- Speaker segment merging

---

## 🧠 Architecture
Audio Input
↓
Whisper (ASR)
↓
WhisperX (Alignment + Diarization)
↓
Reconciler (Merge outputs)
↓
Post-processing
↓
Speaker Matching & Learning
↓
Output (SRT / TXT / JSON)

---

## 📂 Project Structure
.
├── main.py # Main pipeline
├── audio_processor.py # Silence detection & audio processing
├── transcriber_whisper.py # Whisper wrapper
├── transcriber_whisperx.py # WhisperX + diarization
├── reconciler.py # Dual-model merging logic
├── speaker_manager.py # Voiceprint database
├── correction_tool.py # Manual correction workflow
├── config.py # Configuration


---

## ⚙️ Setup

### Requirements

- Python 3.10+
- FFmpeg installed
- CUDA (optional but recommended)

### Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage
Run in web UI
```bash 
python gradio.py
```
Run full pipeline
```bash
python main.py path/to/audio.mp3
```
Whisper only
```bash
python main.py audio.mp3 --only-whisper
```
Silence removal
```bash
python audio_processor.py audio.mp3
```
Apply manual corrections
```bash
python correction_tool.py audio.mp3
```
## 🔁 Speaker Learning Workflow
Run pipeline (initial run)
Edit speaker names in .srt
Apply corrections:
```bash
python correction_tool.py audio.mp3
```
System improves automatically in future runs
## 📊 Output
.srt — subtitles
.txt — readable transcript
.json — structured data
## 🎯 Motivation

Most speech tools focus only on transcription.

This project aims to:

Improve transcription accuracy
Maintain consistent speaker identity
Enable iterative learning from user corrections
## 📌 Future Work
GUI interface
Real-time transcription
API deployment
Cloud integration
## 👨‍💻 Author

Personal project focused on practical system design and real-world problem solving.
