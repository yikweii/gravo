# 🎙️ Speech Transcription App

This is a real-time speech transcription app that:
- Captures microphone input
- Saves audio as WAV
- Transcribes spoken content into text using a HuggingFace Whisper model
- Supports language detection and transcription in Malay, English, and Chinese

---

# Prerequisites
[Python 3.8+](https://www.python.org/downloads/)
pip (Python package installer)

## Required Packages
sounddevice
numpy
torch
transformers
librosa
keyboard
langdetect
fastapi
uvicorn
```
pip install sounddevice numpy torch transformers librosa keyboard langdetect fastapi uvicorn
```

for Whisper dependency, download zip file from [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) or if you have Chocolatey, use this:
```
choco install ffmpeg
```

# Running the app
In your terminal, run:
```
uvicorn fast_api:app --reload
```
When you want to record, run:
```
python mic_capture.py
```
Press q to stop recording.
