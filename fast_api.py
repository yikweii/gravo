from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mic_capture import record_audio, save_audio_as_wav, transcribe_audio

app = FastAPI()
file = "temp_audio.wav"

# run command: uvicorn fast_api:app --reload
@app.post("/record_audio/")
async def record_and_transcribe(duration: int = 5):
    try:
        audio_data = record_audio(duration=duration)
        save_audio_as_wav(audio_data, filename=file)
        transcription = transcribe_audio(file)
        return JSONResponse(content={"transcription": transcription, "message": "Transcription successful!"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})
