# 🎙️ Speech Transcription App

This is a real-time speech transcription app that:
- Captures microphone input
- Saves audio as WAV
- Transcribes spoken content into text using a HuggingFace Whisper model
- Supports language detection and transcription in Malay, English, and Chinese

## Prototype
https://www.figma.com/proto/mHcKzZri13j4wnnvAUW182/UM-Hackathon--Domain-3-?node-id=25-720&t=CH8vKx7FkFbrVDlR-1

## Slides
https://www.canva.com/design/DAGkYQFEhqY/87ytKFx_GfQpsySAZlmNXQ/view?utm_content=DAGkYQFEhqY&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h6029ffcbc7

## Documentation
file:///C:/Users/Lenovo/OneDrive%20-%20Asia%20Pacific%20University/hackathon/Documentation-Group%20Ctrl+Zzz.pdf

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
