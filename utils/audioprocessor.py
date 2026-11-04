import yt_dlp
import os
import re
from urllib.parse import parse_qs, urlparse
from pydub import AudioSegment

DOWNLOAD_DIR='downloads'
CHUNK_CACHE_DIR='cache/audio_chunks'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def extract_video_id(url: str) -> str:
    """Extract the stable video ID from a supported YouTube URL."""
    parsed_url = urlparse(url.strip())
    hostname = parsed_url.netloc.lower().split(":")[0]
    path_parts = [part for part in parsed_url.path.split("/") if part]

    if hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]
        elif path_parts and path_parts[0] in {"embed", "shorts", "live"}:
            video_id = path_parts[1] if len(path_parts) > 1 else None
        else:
            video_id = None
    elif hostname == "youtu.be":
        video_id = path_parts[0] if path_parts else None
    else:
        video_id = None

    if not video_id:
        raise ValueError(f"Could not extract a YouTube video ID from: {url}")

    return video_id


def download_youtube_audio(url :str) ->str:
    video_id = extract_video_id(url)
    cached_audio_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")
    if os.path.exists(cached_audio_path):
        print(f"Using cached audio: {cached_audio_path}")
        return cached_audio_path

    output_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    if not os.path.exists(cached_audio_path):
        raise FileNotFoundError(
            f"yt-dlp completed, but the expected audio file was not created: "
            f"{cached_audio_path}"
        )

    return cached_audio_path


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path:str,chunk_min:int=10)->list:
    source_name = os.path.splitext(os.path.basename(wav_path))[0]
    cache_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name)
    chunk_directory = os.path.join(
        CHUNK_CACHE_DIR,
        f"{cache_name}_{chunk_min}min",
    )
    os.makedirs(chunk_directory, exist_ok=True)
    completion_marker = os.path.join(chunk_directory, ".complete")

    cached_chunks = []
    index = 0
    while True:
        chunk_filename = os.path.join(chunk_directory, f"chunk_{index:03d}.wav")
        if not os.path.isfile(chunk_filename):
            break
        cached_chunks.append(chunk_filename)
        index += 1

    if cached_chunks and os.path.isfile(completion_marker):
        print(f"Using cached audio chunks: {chunk_directory}")
        return cached_chunks

    for filename in os.listdir(chunk_directory):
        if filename.startswith("chunk_") and filename.endswith(".wav"):
            os.remove(os.path.join(chunk_directory, filename))

    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms=chunk_min*60*1000
    chunks=[]
    for i,start in enumerate(range(0,len(audio),chunk_length_ms)):
        chunk=audio[start:start+chunk_length_ms]
        chunk_filename=os.path.join(chunk_directory, f"chunk_{i:03d}.wav")
        chunk.export(chunk_filename,format="wav")
        chunks.append(chunk_filename)
    with open(completion_marker, "w", encoding="utf-8") as marker:
        marker.write("complete\n")
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
