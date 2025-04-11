import sounddevice as sd
import numpy as np
import wave
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
from langdetect import detect
import keyboard
import time
import os
import librosa 


chatbot = pipeline("text2text-generation", model="google/flan-t5-large")

processor = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-medium-v2")
model = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-medium-v2")
t_file = "transcription.txt"  # transcription filename
max_duration = 20 # max record time: 20 sec
samplerate = 16000

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

def get_filename(base_name, extension):
    i = 1
    while os.path.exists(f"{base_name}{i}{extension}"):
        i += 1
    return f"{base_name}{i}{extension}"

def save_audio(audio_data, samplerate, filename=None):
    if filename is None:
        filename = get_filename(base_name="audio", extension=".wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)
    print(f"Audio saved as {filename}")
    return filename

# Language detection
def lang_detect(text):
    try:
        supported_languages = ['en', 'ms', 'zh', 'ta']
        language = detect(text)
        if language in supported_languages:
            return language
        else:
            return 'ms'  # Default to malay
    except Exception as e:
        print(f"Error detecting language: {e}")
        return 'ms'

def transcribe_audio(filename):
    start_time = time.time()
    audio_input, _ = librosa.load(filename, sr=samplerate)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=samplerate)
    attention_mask = torch.ones(inputs['input_features'].shape, dtype=torch.long) 
    with torch.no_grad():
        r = model.generate(inputs['input_features'], attention_mask=attention_mask, language='ms', return_timestamps=True)
    transcription = processor.tokenizer.decode(r[0], skip_special_tokens=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Transcription took: {minutes} min {seconds} sec")
    return transcription

def add_transcript_to_file(transcription, audio_label, t_file):
    with open(t_file, "a", encoding="utf-8") as f:
        f.write(f"{audio_label}\n{transcription}\n\n")


def get_chatbot_response(text, language):
    prompt = (
        f"You are a smart assistant for a Grab driver. The driver speaks in {language}. "
        f"Your job is to understand whether the driver is giving a command or just making conversation. "
        f"Reply appropriately:\n\n"
        
        "If it's a command (like 'call customer', 'pick up food', 'add fuel'), update the task status clearly.\n"
        "If it's casual conversation (like 'I'm tired', 'hello'), respond in a friendly, human way.\n\n"

        "Examples:\n"
        "Driver: arriving in 5 minutes\n"
        "Assistant: Okay, I've sent a message to the customer: You will arrive in 5 minutes.\n\n"
        "Driver: call customer\n"
        "Assistant: Calling the customer now.\n\n"
        "Driver: picked up the food\n"
        "Assistant: Status updated: Order has been picked up.\n\n"
        "Driver: tambah minyak\n"
        "Assistant: Status updated: Driver is adding fuel now.\n\n"
        "Driver: saya ambil makanan\n"
        "Assistant: Status updated: Order has been picked up.\n\n"
        "Driver: hello\n"
        "Assistant: Hi there! Hope your deliveries are going well.\n\n"
        "Driver: penat sangat hari ini\n"
        "Assistant: Dah lama memandu? Rehat sekejap kalau boleh ya.\n\n"
        
        f"Driver: {text}\n"
        "Assistant:"
    )
    result = chatbot(prompt, max_length=100, do_sample=True)[0]['generated_text']
    print(f"🤖 Assistant ({language}): {result}")
    return result


if __name__ == "__main__":
    audio_data = record_audio(max_duration, samplerate)
    if audio_data.size == 0:
        print("No audio recorded.")
    else:
        a_file = get_filename(base_name="audio", extension=".wav")
        audio_filename = save_audio(audio_data, samplerate, filename=a_file)
        transcription = transcribe_audio(audio_filename)
        detected_language = lang_detect(transcription)
        add_transcript_to_file(transcription, f"Transcription for {audio_filename}", t_file)
        print(f"Transcription: {transcription}")
        
        # Get chatbot response in the same language
        get_chatbot_response(transcription, detected_language)

