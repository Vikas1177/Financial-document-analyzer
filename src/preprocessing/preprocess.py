from langchain_text_splitters import RecursiveCharacterTextSplitter
from Langchain.Schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import re

def load_text_from_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_and_embed_risk_factors(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_local: bool = False,
    persist_directory: str = "./risk_factors_embeddings",
    embedding_model: str = "all-MiniLM-L6-v2" 
) -> Chroma:

    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n=== FILING #", 
            "\n\n",          
            "\n",           
            ". ",          
            " ",             
            ""              
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )

    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks")

    documents = []
    for i, chunk in enumerate(chunks):
        filing_num = extract_filing_number(chunk)
        
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "filing_number": filing_num,
                "chunk_size": len(chunk),
                "document_type": "risk_factors"
            }
        )
        documents.append(doc)

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_metadata={"hnsw:space": "cosine"} 
        )
    
    if save_local:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"} 
        )
    
    return vector_store

def load_existing_embeddings(
    persist_directory: str = "./risk_factors_embeddings",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    collection = vector_store._collection
    count = collection.count()
    return vector_store

def extract_filing_number(chunk: str) -> str:
    match = re.search(r'=== FILING #(\d+)', chunk)
    return match.group(1) if match else "unknown"

def query_embeddings(
    vector_store: Chroma, 
    query: str, 
    k: int = 5
) -> list:

    results = vector_store.similarity_search_with_score(query, k=k)
    
    return results
