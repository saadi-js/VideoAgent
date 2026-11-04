import whisper
import os

WHISPER_MODEL =os.getenv("WHISPER_MODEL", "small")  

_model=None

def load_model():
    global _model

    if _model  is None:
        print(f"Loading Whisper model '{WHISPER_MODEL}'...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model

def transcribe_chunk(audio_path: str,translate: bool=True) -> str:
    model = load_model()
    task="translate" if translate else "transcribe"
    result = model.transcribe(audio_path, task=task)

    return result["text"]

def transcribe_all(chunks: list, translate: bool=False) -> str:
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        transcription = transcribe_chunk(chunk, translate=translate)
        full_transcription += transcription + " "
    return full_transcription.strip()



