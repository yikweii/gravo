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
import matplotlib.pyplot as plt
from scipy.io import wavfile
import fir_filter
from scipy.signal import butter, lfilter,filtfilt
import soundfile as sf
from scipy.fft import fft, fftfreq
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
get_speech_timestamps, _, _, _, _ = utils


processor = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-medium-v2")
model = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-medium-v2")
t_file = "transcription.txt"  # transcription filename
max_duration = 20 # max record time: 20 sec
samplerate = 16000

chatbot = pipeline("text2text-generation", model="google/flan-t5-large")


def vad_segment(audio, sample_rate):
    audio_tensor = torch.from_numpy(audio).float()
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)

    speech_segments = []
    noise_segments = []

    last_end = 0
    for ts in speech_timestamps:
        start, end = ts['start'], ts['end']
        if start > last_end:
            noise_segments.append(audio[last_end:start])  # non-speech before this speech
        speech_segments.append(audio[start:end])
        last_end = end

    if last_end < len(audio):
        noise_segments.append(audio[last_end:])  # non-speech after last speech segment

    speech_audio = np.concatenate(speech_segments) if speech_segments else np.array([])
    noise_audio = np.concatenate(noise_segments) if noise_segments else np.array([])

    return speech_audio, noise_audio


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

def save_audio_frommic(audio_data, samplerate, filename=None):
    if filename is None:
        filename = get_filename(base_name="audio", extension=".wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio_data)
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
        
# ====================== FILTER FUNCTIONS ======================
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_butter_filter(audio, fs, lowcut=300.0, highcut=3400.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio


# ====================== UTILITY FUNCTIONS ======================
def save_audio_as_wav(audio_data, samplerate, filename):
    try:
        
        # Always normalize (even low volume)
        peak = np.max(np.abs(audio_data))
        audio_normalized = audio_data / (1.1 * peak) if peak != 0 else audio_data

        # Create output directory if needed
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

        # Save file
        sf.write(filename, audio_normalized, samplerate, subtype='PCM_16')

        return True

    except Exception as e:
        print(f"[ERROR] Failed to save {filename}: {str(e)}")
        return False


# [Previous imports remain the same...]

def processing_sound(input_path):

    # 1. Create output folder
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load audio
    try:
        fs, data = wavfile.read(input_path)

        if len(data.shape) > 1:
            data = data[:, 0]

    except Exception as e:
        return None

    # 3. Save original segment for comparison
    original_path = os.path.join(output_dir, f"original_segment{timestamp}.wav")
    save_audio_as_wav(data, fs, original_path)
    
    # 4. Get transcription of original audio
    try:
        original_transcription = transcribe_audio(original_path)
        detected_language = lang_detect(original_transcription)
        add_transcript_to_file(original_transcription, f"Original Audio {os.path.basename(input_path)}", t_file)
        print(f"Original Transcription: {original_transcription}")
    except Exception as e:
        print(f"[ERROR] Original transcription failed: {e}")

    # 5. Filtering
    try:
        # Voice Activity Detection
        speech_signal, noise_signal = vad_segment(data.astype(np.float32), fs)
        
        # Save segments
        speech_path = os.path.join(output_dir, f"speech_segment{timestamp}.wav")
        noise_path = os.path.join(output_dir, f"noise_segment{timestamp}.wav")
        save_audio_as_wav(speech_signal, fs, speech_path)
        save_audio_as_wav(noise_signal, fs, noise_path)

        _, noise_reference = wavfile.read(noise_path)
        noise_reference = noise_reference.astype(np.float32) / 32768.0

        # Bandpass filter
        filtered_speech = apply_butter_filter(speech_signal, fs)
                # FIR noise cancellation
        # Save final output
        filtered_path = os.path.join(output_dir, f"final_output{timestamp}.wav")
        save_audio_as_wav(filtered_speech, fs, filtered_path)
        
        # Get transcription of filtered audio
        try:
            filtered_transcription = transcribe_audio(filtered_path)
            detected_language = lang_detect(filtered_transcription)
            add_transcript_to_file(filtered_transcription, f"Filtered Audio {os.path.basename(input_path)}", t_file)
            print(f"Filtered Transcription: {filtered_transcription}")
        except Exception as e:
            print(f"[ERROR] Filtered transcription failed: {e}")

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return None

    # 6. Visualization
    try:
        t_full = np.linspace(0, len(data)/fs, len(data))
        t_speech = np.linspace(0, len(speech_signal)/fs, len(speech_signal))
        t_filtered = np.linspace(0, len(filtered_speech)/fs, len(filtered_speech))
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(t_full, data)
        plt.title("Original Signal")
        plt.grid()
        
        plt.subplot(3, 1, 2)
        plt.plot(t_speech, speech_signal)
        plt.title("VAD Speech Segment")
        plt.grid()
        
        plt.subplot(3, 1, 3)
        plt.plot(t_filtered, filtered_speech)
        plt.title("After Bandpass Filter")
        plt.grid()
        

        plot_path = os.path.join(output_dir, f"comparison_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
    except Exception as e:
        print(f"[WARNING] Visualization failed: {e}")

    return filtered_path, filtered_transcription, detected_language

def get_chatbot_response(text, language):
    prompt = (f"""You are a voice assistant for a Grab driver. Respond to the driver's input clearly.
              
        Driver: {text}q
        Assistant:"""
    )
    result = chatbot(prompt, max_length=100, do_sample=True)[0]['generated_text']
    print(f"Assistant ({language}): {result}")
    return result

def match_command(text, lang):
    text = text.lower()

    command_map = {
        ("call customer", "panggil pelanggan"): "Calling the customer now.",
        ("order picked", "ambil makanan", "saya ambil makanan"): "Status updated: Order has been picked up.",
        ("add fuel", "tambah minyak"): "Status updated: Driver is adding fuel now.",
        ("i have arrived", "sudah sampai", "saya sudah sampai"): "Okay, I've sent a message to the customer: You have arrived.",
        ("i will arrive", "akan tiba", "sampai dalam", "dalam 5 minit"): "Okay, I've sent a message to the customer: You will arrive soon.",
        ("accident", "kemalangan"): "Emergency alert: Accident reported. Notifying support.",
        ("traffic jam", "jem"): "Status updated: Driver is currently stuck in traffic.",
        ("emergency record", "rekod kecemasan"): "Recording started."
    }

    for keywords, response in command_map.items():
        if any(keyword in text for keyword in keywords):
            return response
    return None

def handle_driver_input(text, language):
    command_response = match_command(text, language)
    if command_response:
        print(f"Assistant ({language}): {command_response}")

        # Trigger emergency recording if needed
        if "recording started" in command_response.lower():
            emergency_audio = record_audio(max_duration, samplerate)
            if emergency_audio.size > 0:
                filename = save_audio_frommic(emergency_audio, samplerate, filename=get_filename("emergency_audio", ".wav"))
                print(f"Emergency audio recorded and saved as: {filename}")
        return command_response
    else:
        return get_chatbot_response(text, language)

if __name__ == "__main__":
    # Record audio
    audio_data = record_audio(max_duration, samplerate)
    if audio_data.size == 0:
        print("No audio recorded.")
    else:
        # Save recording
        audio_filename = save_audio_frommic(audio_data, samplerate)
        if audio_filename:
            # Process and filter
            result_path, transcription, language = processing_sound(audio_filename)
            if result_path and transcription and language:
                handle_driver_input(transcription, language)
                print(f"\nProcessing complete!")
            else:
                print("\nProcessing failed")
        else:
            print("\nFailed to save recorded audio")


