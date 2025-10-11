import os
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
import chromadb

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence


from src.llm_runner import RiskFactorsLLMRunner
from src.downloader.sec_client import get_10k_urls_for_ticker
from src.scraping.scrape import extract_risk_factors
from src.preprocessing.preprocess import chunk_and_embed_risk_factors, query_embeddings

load_dotenv()

urls = get_10k_urls_for_ticker('MSFT',2)
risk_factors_list = extract_risk_factors(urls,select=[0])

if risk_factors_list and len(risk_factors_list) > 0:
    risk_factors = "\n\n=== COMBINED RISK FACTORS ===\n\n".join(
        [f"=== FILING #{i} RISK FACTORS ===\n{text}" for i, text in enumerate(risk_factors_list) if text]
    )    
    if not risk_factors.strip():
        raise ValueError("No valid risk factors content extracted")
else:
    raise ValueError("No risk factors extracted")

chroma_client = chromadb.Client() 

# vectorize when there is new start or there is change in input features like year, ticker
# this can be done by adding a new bool in diff functions 
def insert_chroma(risk_factors):
    collection = chroma_client.get_or_create_collection("Risk_collection")
    vector_store = chunk_and_embed_risk_factors(risk_factors)
    return vector_store

def initialize_LLM():
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1
    )

    template = """
    Based on the SEC 10-K Risk Factors context provided, answer the user's question comprehensively and accurately.

    First, provide a direct summary of the answer.
    Then, elaborate on the key points using specific details from the risk factors context.
    Finally, conclude with any implications or insights.

    Important: Only use information from the provided context. If the context doesn't contain relevant information, clearly state that the information is not available in the risk factors.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = prompt | llm
    return qa_chain

def query_chroma(vector_store, query):
    retriever_results = query_embeddings(vector_store, query)

    if isinstance(retriever_results, list):
        context = "\n\n".join([doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in retriever_results])
    else:
        context = str(retriever_results)
    
    qa_chain = initialize_LLM()

    response = qa_chain.invoke({
        "context": retriever_results,
        "question": query
    })

    if hasattr(response, 'content'):
        response_text = response.content
    elif isinstance(response, str):
        response_text = response
    else:
        response_text = str(response)

    
    return response_text
        

answer = query_chroma(insert_chroma(risk_factors), "what are the major risk factors for investment in microsoft?")
print(answer)
    
    
    
    


    


