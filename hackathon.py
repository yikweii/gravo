
from audiomentations import Compose, AddGaussianNoise, PitchShift, HighPassFilter, AddBackgroundNoise, TimeStretch, Shift
import soundfile as sf
import librosa
import os

def adjust_noise_volume(noise, volume_factor=2.0):

    return noise * volume_factor


base_path = os.path.dirname(__file__) 
wav_path = os.path.join(base_path, "WAV")
noise_path = os.path.join(base_path, "Noise")
output_path = os.path.join(base_path, "Augmented")

for wav_file in os.listdir(wav_path):
    if not wav_file.endswith(".wav"):
        continue
    wav_full_path = os.path.join(wav_path, wav_file)
    audio, sr = librosa.load(wav_full_path, sr=None)

    for noise_file in os.listdir(noise_path):
        if not noise_file.endswith(".wav"):
            continue
        noise_full_path = os.path.join(noise_path, noise_file)
        noise, _ = librosa.load(noise_full_path, sr=sr)
        
        # Adjust the noise volume (make it louder)
        noise = adjust_noise_volume(noise, volume_factor=2.0)

        augment = Compose([
            AddBackgroundNoise(sounds_path=noise_full_path, p=1.0),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
            Shift(-0.5, 0.5, p=0.4)
        ])

        augmented_audio = augment(samples=audio, sample_rate=sr)

        # Save with clear name format
        new_filename = f"{os.path.splitext(wav_file)[0]}_with_{os.path.splitext(noise_file)[0]}.wav"
        save_path = os.path.join(output_path, new_filename)
        sf.write(save_path, augmented_audio, sr)

        print(f"Saved: {save_path}")