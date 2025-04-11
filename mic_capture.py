import sounddevice as sd
import numpy as np
import wave
import whisper
import keyboard
import time
import os
from transformers import pipeline
from googletrans import Translator


# model: whisper tiny
model = whisper.load_model("tiny")

# Initialize Translator
translator = Translator()

# Load AI Assistant model
chatbot = pipeline("text2text-generation", model="google/flan-t5-large")  # Multilingual T5

# OpenRouter API Key Setup
# OpenRouter API configuration
#API_KEY = "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"  # Replace with your actual key
#API_URL = "https://openrouter.ai/api/v1/chat/completions"

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
    print(f"Detected Language: {result['language']}")  # Display detected language
    return result['text'], result['language']

def add_transcript_to_file(transcription, audio_label, t_file):
    with open(t_file, "a", encoding="utf-8") as f:  
        f.write(f"{audio_label}\n{transcription}\n\n")


def get_chatbot_response(text, language):
    prompt = (
        f"You are a multilingual smart assistant for a Grab driver. The driver speaks in {language}.\n"
        f"You are a smart AI assistant for a Grab driver. The driver said: \"{text}\"\n"
        "Examples:\n"
        "- If driver says 'arriving in 5 minutes', respond: 'Okay, I've sent a message to the customer: You will arrive in 5 minutes.'\n"
        "- If driver says 'call customer', respond: 'Calling the customer now.'\n"
        "- If driver says 'picked up the food', respond: 'Status updated: Order has been picked up.'\n"
        f"Respond naturally in {language}."
    )
    result = chatbot(prompt, max_length=100, do_sample=True)[0]['generated_text']
    print(f"🤖 Assistant ({language}): {result}")
    return result


if __name__ == "__main__":
    audio_data = record_audio(max_duration, samplerate)
    if audio_data.size == 0:
        print("No audio recorded.")
    else:
        a_file = get_next_available_filename(base_name="audio", extension=".wav")
        audio_filename = save_audio_as_wav(audio_data, samplerate, filename=a_file)
        transcription, language = transcribe_audio(audio_filename)
        add_transcript_to_file(transcription, f"Transcription for {audio_filename}", t_file)
        print(f"Transcription: {transcription}")
        
        # Get chatbot response in the same language
        get_chatbot_response(transcription, language)

