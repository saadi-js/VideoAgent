#ActionableItems,decisions,questions

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts  import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os

def getllm():
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", api_key=api_key,temperature=0.2)

def build_chain(system_prompt:str):
    llm=getllm()

    return(RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{text}")]
    ) | llm | StrOutputParser())

def extract_actionable_items(transcript:str)->str:
    system_prompt = (
        "You are an expert meeting summarizer. Extract all actionable items from the meeting transcript. "
        "Provide a concise list of actionable items, each with a clear description and assigned owner if mentioned. "
        "If no actionable items are found, respond with 'No actionable items found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_decisions(transcript:str)->str:
    system_prompt = (
        "You are an expert meeting summarizer. Extract all decisions made during the meeting from the transcript. "
        "Provide a concise list of decisions, each with a clear description and any relevant context. "
        "If no decisions are found, respond with 'No decisions found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_questions(transcript:str)->str:
    system_prompt = (
        "You are an expert meeting summarizer. Extract all questions raised during the meeting from the transcript. "
        "Provide a concise list of questions, each with a clear description and any relevant context. "
        "If no questions are found, respond with 'No questions found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke(transcript)



