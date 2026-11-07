from dotenv import load_dotenv
import os
import re
from utils.audioprocessor import process_input
from utils.audioprocessor import extract_video_id
from core.transcriber import WHISPER_MODEL, transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question



load_dotenv()  # Load environment variables from .env file
TRANSCRIPT_DIR = "transcripts"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def get_transcript_cache_path(source: str, translate: bool = True) -> str:
    cache_key = get_source_cache_key(source)
    translation_mode = "translate" if translate else "transcribe"
    filename = f"{cache_key}_{WHISPER_MODEL}_{translation_mode}.txt"
    return os.path.join(TRANSCRIPT_DIR, filename)


def get_source_cache_key(source: str) -> str:
    if source.startswith(("http://", "https://")):
        return extract_video_id(source)
    else:
        return re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            os.path.splitext(os.path.basename(source))[0],
        )


def get_or_create_transcript(chunks: list, source: str, translate: bool = True) -> str:
    transcript_path = get_transcript_cache_path(source, translate)
    if os.path.isfile(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as transcript_file:
            transcript = transcript_file.read().strip()
        if transcript:
            print(f"Using cached transcript: {transcript_path}")
            return transcript

    transcript = transcribe_all(chunks, translate=translate)
    temporary_path = f"{transcript_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as transcript_file:
        transcript_file.write(transcript)
    os.replace(temporary_path, transcript_path)
    return transcript

def run_pipeline(source:str)->dict:
    print("Starting AI Vedio Assistant")

    chunks=process_input(source)

    transcript=get_or_create_transcript(chunks, source, translate=True)

    print(f"Raw transcript generated. {transcript[:300]}")

    title=generate_title(transcript)

    summary=summarize(transcript)

    action_item=extract_actionable_items(transcript)

    decisions=extract_decisions(transcript)

    questions=extract_questions(transcript)

    rag_chain=build_rag_chain(transcript, get_source_cache_key(source))

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    result = run_pipeline(source)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['decisions']}")
    print(f"\n❓ Open Questions:\n{result['questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    session_id = get_source_cache_key(source)
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question, session_id=session_id)
        print(f"\n🤖 Assistant: {answer}\n")
