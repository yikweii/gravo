import sounddevice as sd
import numpy as np
import wave
import whisper
import keyboard
import time
import os

# model: whisper tiny
model = whisper.load_model("tiny")

t_file="transcription.txt" #transcription filename
max_duration = 60
samplerate=16000

def record_audio(max_duration, samplerate):
    print(f"Recording...(Press 'q' to stop)")
    audio_buffer = []
    start_time = time.time()
    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        audio_buffer.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16', callback=callback):
        while True:
            elapsed = time.time() - start_time
            if keyboard.is_pressed('q') or elapsed >= max_duration:
                break
            time.sleep(0.1)  # Reduce CPU usage

    print("Recording stopped.")
    audio = np.concatenate(audio_buffer, axis=0).flatten()
    return audio

def get_next_available_filename(base_name="audio", extension=".wav"):
    i = 1
    while os.path.exists(f"{base_name}{i}{extension}"):
        i += 1
    return f"{base_name}{i}{extension}"

def save_audio_as_wav(audio_data, samplerate, filename):
    if filename is None:
        filename = get_next_available_filename()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)
    print(f"Audio saved as {filename}")
    return filename

def transcribe_audio(filename):
    result = model.transcribe(filename, language="ms", fp16=False)  # Malay
    print(f"Whisper Transcription Result: {result}")  # Debugging output
    return result['text']

def add_transcript_to_file(transcription, audio_label, t_file):
    with open(t_file, "a", encoding="utf-8") as f:  
        f.write(f"{audio_label}\n{transcription}\n\n")

if __name__ == "__main__":
    audio_data = record_audio(max_duration, samplerate)
    if audio_data.size == 0:
        print("No audio recorded.")
    else:
        a_file = get_next_available_filename(base_name="audio", extension=".wav")
        audio_filename = save_audio_as_wav(audio_data, samplerate, filename=a_file)
        transcription = transcribe_audio(audio_filename)
        add_transcript_to_file(transcription, f"Transcription for {audio_filename}", t_file)
        print(f"Transcription: {transcription}")

