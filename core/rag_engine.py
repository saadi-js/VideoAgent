from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts  import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from core.vectorstore import (
    build_vector_store,
    load_vector_store,
    get_retriever,
    vector_store_exists,
)


import os

_chat_histories: dict[str, InMemoryChatMessageHistory] = {}

def getllm():
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", api_key=api_key,temperature=0.3)

def format_docs(docs):
    return "\n\n".join([f"\n{doc.page_content}" for doc in docs])


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _chat_histories:
        _chat_histories[session_id] = InMemoryChatMessageHistory()
    return _chat_histories[session_id]


def _create_chat_rag_chain(retriever, llm):
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Rewrite the user's latest question as a standalone question
using the conversation history when necessary. Do not answer the question.
If the question is already standalone, return it unchanged.""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            standalone_question=rewrite_chain,
        )
        | {
            "context": itemgetter("standalone_question")
            | retriever
            | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


def build_rag_chain(transcript: str, source_key: str = "default"):
    if vector_store_exists(source_key, transcript):
        print(f"Using cached vector store: {source_key}")
        vector_store = load_vector_store(source_key)
    else:
        vector_store = build_vector_store(transcript, source_key)

    return _create_chat_rag_chain(get_retriever(vector_store, k=4), getllm())

def load_rag_chain(source_key: str = "default"):
    vector_store=load_vector_store(source_key)
    return _create_chat_rag_chain(get_retriever(vector_store, k=4), getllm())


def ask_question(rag_chain, question: str, session_id: str = "default") -> str:
    response = rag_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    if not isinstance(response, str):
        raise TypeError(f"Expected a string response from the RAG chain, got {type(response)!r}")
    return response






    
