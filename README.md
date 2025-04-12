# 🎙️ Speech Transcription App

This is a real-time speech transcription app that:
- Captures microphone input
- Saves audio as WAV
- Transcribes spoken content into text using a HuggingFace Whisper model
- Supports language detection and transcription in Malay, English, and Chinese

## Prototype
https://www.figma.com/proto/mHcKzZri13j4wnnvAUW182/UM-Hackathon--Domain-3-?node-id=25-720&t=CH8vKx7FkFbrVDlR-1

## Slides
https://cloudmails-my.sharepoint.com/:b:/g/personal/tp070004_mail_apu_edu_my/EfDpngV-eSpOhHdyHxL-g94B6YatiNqnBbI7D1AJ2pR2RA?e=OBJsM4

## Documentation
https://cloudmails-my.sharepoint.com/:b:/g/personal/tp070004_mail_apu_edu_my/EWxfArYJKwpKhIpcEWhnOHQB75iOaKWNQckeYdk-XJNqIg?e=fW8nEE

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
