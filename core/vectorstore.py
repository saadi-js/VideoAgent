import hashlib
import json
import os
import re
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "vector_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "meeting_transcript")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
VECTORSTORE_VERSION = 1


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def get_vector_store_paths(source_key: str) -> tuple[str, str, str, str]:
    safe_key = re.sub(r"[^A-Za-z0-9._-]+", "_", source_key)
    persist_directory = os.path.join(CHROMA_DIR, safe_key)
    collection_name = f"{COLLECTION_NAME}_{safe_key}"
    completion_marker = os.path.join(persist_directory, ".complete")
    metadata_path = os.path.join(persist_directory, "metadata.json")
    return persist_directory, collection_name, completion_marker, metadata_path


def get_transcript_hash(transcript: str) -> str:
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def get_vector_store_metadata(transcript: str) -> dict[str, str | int]:
    return {
        "version": VECTORSTORE_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "transcript_hash": get_transcript_hash(transcript),
    }


def vector_store_exists(source_key: str, transcript: str | None = None) -> bool:
    _, _, completion_marker, metadata_path = get_vector_store_paths(source_key)
    if not os.path.isfile(completion_marker) or not os.path.isfile(metadata_path):
        return False
    if transcript is None:
        return True

    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError):
        return False

    return metadata == get_vector_store_metadata(transcript)


def build_vector_store(transcript: str, source_key: str = "default") -> Chroma:
    print("Building vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(transcript)

    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]

    persist_directory, collection_name, completion_marker, metadata_path = get_vector_store_paths(
        source_key
    )
    if os.path.isdir(persist_directory):
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    temporary_metadata_path = f"{metadata_path}.tmp"
    with open(temporary_metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(get_vector_store_metadata(transcript), metadata_file, indent=2)
    os.replace(temporary_metadata_path, metadata_path)
    with open(completion_marker, "w", encoding="utf-8") as marker:
        marker.write("complete\n")
    return vector_store


def load_vector_store(source_key: str = "default") -> Chroma:
    persist_directory, collection_name, _, _ = get_vector_store_paths(source_key)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
