from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts  import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from core.vectorstore import build_vector_store, load_vector_store, get_retriever


import os

def getllm():
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", api_key=api_key,temperature=0.3)

def format_docs(docs):
    return "\n\n".join([f"\n{doc.page_content}" for doc in docs])


def build_rag_chain(transcript:str):
    vector_store=build_vector_store(transcript)

    retriever=get_retriever(vector_store,k=4)

    llm=getllm()

    prompt = ChatPromptTemplate.from_messages(

        [(
            "system",
            """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}""",
        ),
        ("human", "{question}"),
    ]
    )


    # full LCEL Rag Pipeline

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs), 
            "question": RunnablePassthrough()
        }

        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    vector_store=load_vector_store()

    retriever=get_retriever(vector_store,k=4)

    llm=getllm()

    prompt = ChatPromptTemplate.from_messages(
    
            [(
                "system",
                """You are an expert meeting assistant. Answer the user's question 
    based ONLY on the meeting transcript context provided below.
    
    If the answer is not found in the context, say: 
    "I could not find this information in the meeting transcript."
    
    Always be concise and precise. If quoting someone, mention it clearly.
    
    Context from meeting transcript:
    {context}""",
            ),
            ("human", "{question}"),
        ]
        )

    rag_chain = (
        {
            "context":  retriever| RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question:str)->str:
    return rag_chain.invoke(question)






    
