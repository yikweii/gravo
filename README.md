# Installation guide

## Prerequisites
Python 3.8 or higher
pip (Python package installer)

## Required Packages
```
pip install fastapi uvicorn
pip install sounddevice
pip install numpy
pip install keyboard
pip install openai-whisper
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
