# VoiceScript — Speaker-Aware Speech Transcription System

VoiceScript is a modular Python system for **speech transcription, speaker diarization, and speaker identity learning**.

Unlike standard transcription tools, VoiceScript focuses on:

* **consistent speaker labeling across recordings**
* **correction-driven learning workflow**
* **reusable speaker identity (voiceprint database)**

---

## Demo

*(Add a screenshot or GIF of the Gradio UI here — highly recommended)*

---

## Quick Start (Recommended)

Run the web interface:

```bash
python gradio_app.py
```

Then open the local URL in your browser.

You can:

* Upload an audio file
* Generate speaker-labeled transcripts
* Export SRT / TXT / JSON outputs

---

## Key Features

* **Dual-model transcription** using Whisper and WhisperX
* **Speaker diarization with identity persistence**
* **Custom reconciliation layer** to merge model outputs
* **Voiceprint database** for consistent speaker recognition
* **Manual correction workflow (SRT-based)** for iterative improvement
* **Silence detection and audio preprocessing (FFmpeg)**
* **Multiple output formats** (SRT, TXT, JSON)
* **Environment version tracking for reproducibility**

---

## System Architecture

```text
Audio Input
   ↓
Whisper (timestamp accuracy)
   ↓
WhisperX (alignment + diarization)
   ↓
Reconciler (merge + conflict handling)
   ↓
Speaker Manager (identity mapping + learning)
   ↓
Output (SRT / TXT / JSON)
```

Main pipeline implementation: 

---

## Core Modules

* `main.py` — pipeline orchestrator
* `transcriber_whisper.py` — Whisper transcription
* `transcriber_whisperx.py` — WhisperX alignment & diarization
* `reconciler.py` — dual-model merging logic
* `speaker_manager.py` — voiceprint database & speaker matching
* `correction_tool.py` — manual correction workflow
* `audio_processor.py` — silence detection & audio trimming

Example module (audio preprocessing): 

---

## Workflow

### 1. First Run (Cold Start)

```bash
python main.py path/to/audio.mp3
```

* System generates temporary speaker labels (e.g., SPEAKER_00)
* A mapping file is created

---

### 2. Assign Speaker Names

```bash
python speaker_setup.py apply
```

* Replace temporary labels with real names
* Initialize speaker database

---

### 3. Improve Accuracy (Optional)

Edit the generated `.srt` file and correct speaker names, then run:

```bash
python correction_tool.py path/to/audio.mp3
```

* Updates speaker embeddings
* Improves future recognition

Correction workflow: 

---

## CLI Usage (Advanced)

Full pipeline:

```bash
python main.py path/to/audio.mp3
```

Whisper only:

```bash
python main.py path/to/audio.mp3 --only-whisper
```

WhisperX only:

```bash
python main.py path/to/audio.mp3 --only-whisperx
```

---

## Technical Highlights

* Designed a **modular Python architecture** with independent components
* Implemented **dual-model reconciliation logic** for improved accuracy
* Built a **speaker identity system using embeddings and similarity matching**
* Developed a **correction feedback loop** to iteratively improve results
* Integrated **FFmpeg-based audio preprocessing**
* Created a **persistent speaker database with weighted embeddings**

Speaker system implementation: 

---

## Project Scope

This is a **personal engineering project** focused on:

* solving real-world transcription limitations
* building reusable speech-processing tools
* demonstrating system design and backend architecture

It is designed for **local usage and experimentation**, not production deployment.

---

## Requirements

* Python 3.10+
* FFmpeg
* GPU recommended (CUDA)

---

## Notes

* First run requires manual speaker labeling
* Performance depends on hardware and audio quality
* Large models may require significant VRAM

---

## Author

Personal project for learning and practical problem solving in:

* Python backend development
* AI/audio processing pipelines
* system design and tooling
