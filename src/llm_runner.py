import os
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class RiskFactorsLLMRunner:
    """
    LLM Runner that uses a pre-built ChromaDB vector store (local embeddings)
    and the FREE Google Gemini API for generation.
    """

    def __init__(
        self,
        chunks_directory: str,
        persist_directory: str = "./risk_factors_embeddings",
        llm_model: str = "gemini-2.0-flash"
    ):
        """
        Args:
            chunks_directory: Path to directory of .txt chunk files (unused here).
            persist_directory: Path where ChromaDB embeddings were persisted.
            llm_model: Google Gemini model name.
        """
        self.chunks_directory = Path(chunks_directory)
        self.persist_directory = persist_directory

        # Load Google API key
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        # Initialize Google LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=self.api_key,
            temperature=0.1
        )

        # Vector store and retriever placeholders
        self.vector_store: Optional[Chroma] = None
        self.retriever = None
        self.chain = None

        print(f"✓ Runner initialized using persisted embeddings at '{persist_directory}'.")

    def load_vector_store(self) -> None:
        """
        Load the existing ChromaDB vector store from disk,
        providing the embedding function so queries can be embedded.
        """
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"Persist directory not found: {self.persist_directory}")

        # Reinstantiate the same free local embedding model used initially
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Load Chroma with embedding function for query embeddings
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        print("✓ Loaded vector store with embedding function and initialized retriever.")


    def setup_qa_chain(self, custom_template: Optional[str] = None) -> None:
        """
        Set up the retrieval-augmented QA chain with a default or custom prompt.
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized. Call load_vector_store() first.")

        default_template = """
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
        prompt = PromptTemplate.from_template(custom_template or default_template)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("✓ QA chain initialized.")

    def query(self, question: str) -> str:
        """
        Query the RAG chain with a user question.
        """
        if not self.chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        return self.chain.invoke(question)

    def interactive_session(self) -> None:
        """
        Start a simple REPL for question-answering.
        """
        print("\n=== SEC Risk Factors Q&A (Free Google LLM) ===")
        print("Type 'exit' or 'quit' to end the session.\n")

        while True:
            user_q = input("Question: ").strip()
            if user_q.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if not user_q:
                continue
            print("\nAnswer:\n" + self.query(user_q) + "\n")

def main():
    CHUNKS_DIR = "data/chunked_risk_factors"            # existing chunk dir (not re-used here)
    PERSIST_DIR = "./risk_factors_embeddings"           # where embeddings were saved

    runner = RiskFactorsLLMRunner(CHUNKS_DIR, PERSIST_DIR)
    runner.load_vector_store()
    runner.setup_qa_chain()

    # Example usage
    for q in [
        "What are the main cybersecurity risks mentioned?",
        "How does economic uncertainty affect the business?"
    ]:
        print(f"\nQ: {q}\nA: {runner.query(q)}\n{'-'*50}")

    # Enter interactive mode
    runner.interactive_session()

if __name__ == "__main__":
    main()
