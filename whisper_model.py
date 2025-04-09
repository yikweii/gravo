import whisper
import os

def transcribe_audio(audio_path, output_txt="transcription.txt"):
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path)
        transcribed_text = result["text"].strip()
        
        if not transcribed_text:
            print("No clear speech detected – likely noise.")
            transcribed_text = "[NOISE]"
        else:
            print("Detected speech:", transcribed_text)
        
        # Save to text file
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        print(f"Transcription saved to {os.path.abspath(output_txt)}")
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage
audio_path = "noise.wav"
transcribe_audio(audio_path)