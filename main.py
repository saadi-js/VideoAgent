from dotenv import load_dotenv
import os
from utils.audioprocessor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_actionable_items, extract_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question



load_dotenv()  # Load environment variables from .env file

def run_pipeline(source:str)->dict:
    print("Starting AI Vedio Assistant")

    chunks=process_input(source)

    transcript=transcribe_all(chunks,translate=True)

    print(f"Raw transcript generated. {transcript[:100]}")

    title=generate_title(transcript)

    summary=summarize(transcript)

    action_item=extract_actionable_items(transcript)

    decisions=extract_decisions(transcript)

    questions=extract_questions(transcript)

    rag_chain=build_rag_chain(transcript)

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
    language = input("Language (english/hinglish): ").strip() or "english"
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
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")



