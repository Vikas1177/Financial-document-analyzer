from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import re

def load_text_from_file(file_path: str) -> str:
    """Load text content from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_and_embed_risk_factors(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    save_local: bool = False,
    persist_directory: str = "./risk_factors_embeddings",
    embedding_model: str = "all-MiniLM-L6-v2"  # Fast, free, local model
) -> Chroma:
    """
    Chunk risk factors text and save as vector embeddings using FREE local model.
    
    Args:
        file_path: Path to the risk factors text file
        chunk_size: Maximum size of each chunk (characters)
        chunk_overlap: Overlap between chunks (characters)
        persist_directory: Directory to save ChromaDB embeddings
        embedding_model: FREE local embedding model name
    
    Returns:
        ChromaDB vector store with embedded chunks
    """
    
    
    # Initialize the text splitter with optimized parameters for SEC filings
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n=== FILING #",  # Filing separators
            "\n\n",           # Paragraph breaks
            "\n",             # Line breaks
            ". ",             # Sentence endings
            " ",              # Word boundaries
            ""                # Character level
        ],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        keep_separator=True
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks")
    
    # Create Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        # Extract filing number from chunk if present
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
    
    # Initialize FREE local embeddings (no API calls needed)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_metadata={"hnsw:space": "cosine"}  # Optimized for similarity search
        )
    
    if save_local:
        # Create persistent directory
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Create ChromaDB vector store and save embeddings to disk
        print(f"💾 Creating vector embeddings and saving to {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}  # Optimized for similarity search
        )
        
        print(f"✅ Successfully saved {len(documents)} embedded chunks to disk!")
        print(f"📁 Embeddings stored in: {persist_directory}")
    
    return vector_store

def load_existing_embeddings(
    persist_directory: str = "./risk_factors_embeddings",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Chroma:
    """
    Load existing vector embeddings from disk (FAST - no re-embedding needed).
    
    Args:
        persist_directory: Directory containing saved embeddings
        embedding_model: Same model used during creation
    
    Returns:
        ChromaDB vector store loaded from disk
    """
    
    print(f"📂 Loading existing embeddings from {persist_directory}")
    
    # Initialize the same embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Load existing vector store from disk
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    # Get collection stats
    collection = vector_store._collection
    count = collection.count()
    
    print(f"✅ Loaded {count} embedded chunks from disk")
    return vector_store

def extract_filing_number(chunk: str) -> str:
    """Extract filing number from chunk text."""
    match = re.search(r'=== FILING #(\d+)', chunk)
    return match.group(1) if match else "unknown"

def query_embeddings(
    vector_store: Chroma, 
    query: str, 
    k: int = 5
) -> list:
    """
    Query the vector embeddings for similar chunks.
    
    Args:
        vector_store: ChromaDB vector store
        query: Search query
        k: Number of results to return
    
    Returns:
        List of similar document chunks
    """
    
    print(f"🔍 Searching for: '{query}'")
    
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=k)
    
    return results

# Example usage and testing
if __name__ == "__main__":
    
    # Configuration
    RISK_FACTORS_FILE = 'data/risk_factors_filings.txt'
    EMBEDDINGS_DIR = './risk_factors_embeddings'
    
    # Check if embeddings already exist
    if Path(EMBEDDINGS_DIR).exists():
        print("🔄 Found existing embeddings, loading from disk...")
        vector_store = load_existing_embeddings(EMBEDDINGS_DIR)
    else:
        print("🆕 Creating new embeddings...")
        vector_store = chunk_and_embed_risk_factors(
            text=load_text_from_file(RISK_FACTORS_FILE),
        )
    
    # # Test queries
    # sample_queries = [
    #     "cybersecurity risks",
    #     "regulatory compliance",
    #     "economic uncertainty",
    #     "supply chain disruptions",
    #     "market volatility"
    # ]
    
    # print("\n" + "="*60)
    # print("🧪 TESTING VECTOR SEARCH")
    # print("="*60)
    
    # for query in sample_queries:
    #     results = query_embeddings(vector_store, query, k=3)
    #     print("-" * 60)
    
    # print(f"\n💡 Next time you run this, embeddings will load instantly from {EMBEDDINGS_DIR}")
