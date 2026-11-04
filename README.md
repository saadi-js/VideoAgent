# VidAgent

VidAgent is an AI meeting assistant that downloads or reads meeting audio, converts it into speech, translates it into English when requested, and uses Gemini with retrieval-augmented generation (RAG) to answer questions about the meeting.

## Technology

- **yt-dlp** downloads audio from YouTube.
- **FFmpeg** converts and post-processes audio and video files.
- **OpenAI Whisper** performs local speech-to-text transcription.
- **Ollama `nomic-embed-text`** creates local text embeddings.
- **Chroma** stores and retrieves transcript embeddings.
- **Google Gemini** generates titles, summaries, extracted meeting information, and answers to questions.
- **Streamlit** and related packages are available for the UI layer.

## Requirements

- Python 3.10 or newer
- FFmpeg and `ffprobe` installed and available on `PATH`
- Ollama installed and running
- Ollama's embedding model:

  ```powershell
  ollama pull nomic-embed-text
  ```

- A Gemini API key in `.env`:

  ```env
  GEMINI_API_KEY="your-api-key"
  WHISPER_MODEL="small"
  ```

Install the Python dependencies with:

```powershell
uv pip install -r requirement.txt
```

## Original application flow

This section describes how the program worked before cache behavior was added.

### 1. Start the application

Run the application from the project root so Python can resolve both the `core` and `utils` packages:

```powershell
cd C:\Users\user\OneDrive\Desktop\VidAgent
& .\.venv\Scripts\python.exe main.py
```

The program asks for:

```text
Enter YouTube URL or local file path:
```

### 2. Acquire the audio

The input is passed to `process_input()` in `utils/audioprocessor.py`.

For a YouTube URL:

1. `yt-dlp` selects the best available audio.
2. The audio is downloaded into `downloads/`.
3. FFmpeg converts the downloaded stream to WAV.

For a local file:

1. The input file is opened with `pydub`.
2. FFmpeg decodes the audio or video format.
3. The file is converted to mono, 16 kHz WAV audio.

### 3. Split the audio

The WAV file is split into chunks, with a default duration of ten minutes per chunk. The chunks are then passed to the transcription layer.

### 4. Transcribe with Whisper

`core/transcriber.py` loads the Whisper model selected by `WHISPER_MODEL`. Each audio chunk is transcribed locally.

The main pipeline currently requests English translation:

```python
transcript = transcribe_all(chunks, translate=True)
```

Whisper returns the combined transcript text.

### 5. Generate meeting information with Gemini

The transcript is passed to Gemini through the functions in `core/summarize.py` and `core/extractor.py`.

The application generates:

- A short meeting title
- A professional summary
- Actionable items
- Decisions
- Open questions

### 6. Build the vector store

The transcript is passed to `build_rag_chain()` in `core/rag_engine.py`.

The vector-store flow is:

1. `RecursiveCharacterTextSplitter` divides the transcript into chunks.
2. Each chunk becomes a LangChain `Document`.
3. Ollama's local `nomic-embed-text` model creates an embedding for each document.
4. Chroma stores the documents and embeddings in `vector_db/`.
5. A similarity retriever is created for the RAG chain.

### 7. Ask questions about the meeting

The application enters an interactive question loop:

```text
You:
```

For every question:

1. The retriever searches Chroma for the most relevant transcript chunks.
2. The retrieved chunks are formatted as context.
3. The question and context are sent to Gemini.
4. Gemini returns an answer based only on the meeting context.

Type `exit`, `quit`, or `q` to leave the question loop.

## Cache improvements

The application now avoids repeating expensive work when the same source is processed again.

### Stable source keys

YouTube titles are not used as cache identifiers because titles can change and may contain invalid filename characters. The application extracts the stable YouTube video ID from the URL.

For example:

```text
https://www.youtube.com/watch?v=g6Die8gVFn0
```

uses:

```text
g6Die8gVFn0
```

Local files use a sanitized version of the filename as the source key.

### Audio cache

Downloaded YouTube audio is stored using the video ID:

```text
downloads/
└── g6Die8gVFn0.wav
```

When the same URL is entered again:

1. The video ID is extracted.
2. `downloads/<video_id>.wav` is checked.
3. If the file exists, it is reused.
4. Otherwise, yt-dlp downloads and converts the audio.

### Audio-chunk cache

Generated audio chunks are stored in a source-specific directory:

```text
cache/
└── audio_chunks/
    └── g6Die8gVFn0_10min/
        ├── chunk_000.wav
        ├── chunk_001.wav
        └── .complete
```

The `.complete` marker is written only after every chunk is generated successfully. This prevents a partially generated directory from being treated as a valid cache.

### Transcript cache

Transcripts are stored in `transcripts/` using the source key, Whisper model, and transcription mode:

```text
transcripts/
└── g6Die8gVFn0_small_translate.txt
```

Before calling Whisper, the application checks for this file. A non-empty cached transcript is reused. If it does not exist, Whisper runs and the new transcript is saved.

The transcript cache includes the Whisper model and mode so that changing from one model or mode to another creates a separate cache entry.

### Per-source vector stores

Each source has its own Chroma directory and collection:

```text
vector_db/
└── g6Die8gVFn0/
    ├── chroma.sqlite3
    ├── metadata.json
    └── .complete
```

The collection name includes the source key, for example:

```text
meeting_transcript_g6Die8gVFn0
```

This prevents vectors from different meetings from being mixed together.

### Embedding metadata validation

Each vector store records metadata including:

- Embedding model
- Chunk size
- Chunk overlap
- Transcript SHA-256 hash
- Vector-store metadata version

When the same source is processed again, the application compares the current configuration and transcript hash with `metadata.json`.

If everything matches:

```text
Using cached vector store: g6Die8gVFn0
```

The existing Chroma store is loaded and Ollama does not regenerate the embeddings.

If the transcript or embedding configuration changes:

1. The old source-specific vector store is removed.
2. The transcript is split again.
3. New embeddings are generated.
4. New metadata is saved.
5. The `.complete` marker is written after successful completion.

### Current repeated-URL flow

On the first run:

```text
URL
→ download audio
→ create audio chunks
→ transcribe with Whisper
→ generate Gemini meeting analysis
→ create Ollama embeddings
→ save Chroma vector store
→ answer questions
```

On later runs with the same URL and compatible configuration:

```text
URL
→ reuse downloaded audio
→ reuse audio chunks
→ reuse transcript
→ generate Gemini meeting analysis
→ load existing Chroma embeddings
→ answer questions
```

The current cache prevents duplicate audio downloads, duplicate Whisper transcription, and duplicate embedding generation. Gemini title, summary, action-item, decision, and question generation still runs each time because those results are not cached yet.

## Generated files

The following directories are generated locally and are excluded from Git:

```text
downloads/
transcripts/
vector_db/
cache/
```

Do not commit `.env`, API keys, downloaded media, generated transcripts, or local vector databases.
