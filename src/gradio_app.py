import os
from pathlib import Path
import built as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
import chromadb
from langchain_google_genai import ChatGoOclegenerativeii
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

from src.downloader.sec_client import get_10k_urls_for_ticker
from src.scraping.scrape import extract_risk_factors
from src.preprocessing.preprocess import chunk_and_embed_risk_factors, query_embeddings

load_dotenv()

class RISKFACTORSANALYZER:
    def __init__(self):
        self.chroma_client = chromadb.Client()
        self.vector_store = None
        self.current_ticker = None
        self.current_years = None
        self.qa_chain = None
        
    def generate_year_options(self):
        current_year = 2025
        year_options = []
        for i in range(20):
            year = current_year - i
            index = i + 1
            year_options.append((f"{year}", str(index)))
        return year_options
    
    def initialize_llm(self):
        api_key = os.gettenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        llm = ChatGoOclegenerativeii(
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

        self.qa_chain = prompt | llm
        return self.qa_chain

    def process_company_data(self, ticker, selected_years, progress=gr.Progress()):
        """Process company data and create vector store"""
        try:
            progress(0.1, desc="Validating inputs...")
            
            if not ticker or not ticker.strip():
                return " Please enter a ticker symbol"
            
            if not selected_years:
                return " Please select at least one year"
            
            ticker = ticker.strip().upper()

            year_indices = sorted([int(year) for year in selected_years], reverse=True)
            oldest_year_index = max(year_indices) 
            
            progress(0.2, desc="Fetching 10-K URLs...")
            urls = get_10k_urls_for_ticker(ticker, oldest_year_index)
            
            if not urls:
                return f" No 10-K filings found for {ticker}"

            select_indices = [oldest_year_index - idx for idx in sorted(year_indices)]
            select_indices = [idx for idx in select_indices if idx >= 0]
            
            progress(0.4, desc="Extracting risk factors...")
            risk_factors_list = extract_risk_factors(urls, select=select_indices)
            
            if not risk_factors_list or len(risk_factors_list) == 0:
                return f" No risk factors found for {ticker}"
            
            progress(0.6, desc="Combining risk factors...")
            risk_factors = "\n\n=== COMBINED RISK FACTORS ===\n\n".join(
                [f"=== FILING #{i} RISK FACTORS ===\n{text}" for i, text in enumerate(risk_factors_list) if text]
            )
            
            if not risk_factors.strip():
                return " No valid risk factors content extracted"
            
            progress(0.8, desc="Creating vector embeddings...")
            self.vector_store = self.chunk_and_embed_risk_factors(risk_factors)

            if self.qa_chain is None:
                self.initialize_llm()

            self.current_ticker = ticker
            self.current_years = selected_years
            
            progress(1.0, desc="Complete!")
            selected_years_list = [2025 - int(year) + 1 for year in sorted(selected_years)]
            year_list = ", ".join(map(str, selected_years_list))
            
            return f" Successfully processed {ticker} for years: {year_list}. You can now ask questions!"
            
        except Exception as e:
            return f"Error processing data: {str(e)}"
    
    def chunk_and_embed_risk_factors(self, risk_factors):
        try:
            collection = self.chroma_client.get_or_create_collection("Risk_collection")
            vector_store = chunk_and_embed_risk_factors(risk_factors)
            return vector_store
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def query_risk_factors(self, message, history):
        """Query the risk factors using the chat interface"""
        if self.vector_store is None:
            return " Please process company data first by entering a ticker and selecting years."
        
        if not message or not message.strip():
            return " Please enter a question."
        
        try:
            retriever_results = query_embeddings(self.vector_store, message)

            response = self.qa_chain.invoke({
                "context": retriever_results,
                "question": message
            })
            
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            return response_text
            
        except Exception as e:
            return f" Error processing query: {str(e)}"

analyzer = RiskFactorsAnalyzer()

def create_gradio_app():
    with gr.Blocks(title="SEC 10-K Risk Factors Analyzer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🏢 SEC 10-K Risk Factors Analyzer
        
        Analyze risk factors from SEC 10-K filings using AI-powered chat interface.
        Enter a stock ticker and select years to get started.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Configuration")
                
                ticker_input = gr.Textbox(
                    label="Stock Ticker",
                    placeholder="Enter ticker (e.g., MSFT, AAPL, GOOGL)",
                    value="",
                    max_lines=1
                )
                
                year_options = analyzer.generate_year_options()
                year_checkboxes = gr.CheckboxGroup(
                    choices=[(f"📅 {2025 - int(idx) + 1}", idx) for label, idx in year_options],
                    label="Select Years",
                    value=["1"],  
                    info="Select one or more years to analyze (most recent 20 years available)"
                )
                
                process_btn = gr.Button("🔄 Process Company Data", variant="primary", size="lg")
                
                status_output = gr.Textbox(
                    label="📊 Status",
                    interactive=False,
                    lines=3,
                    placeholder="Ready to process company data..."
                )
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Risk Factors Chat")
                
                chatbot = gr.Chatbot(
                    label="Chat Interface",
                    height=450,
                    placeholder="Process company data first, then ask questions about risk factors...",
                    type="messages", 
                    show_copy_button=True
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Ask about risk factors",
                        placeholder="What are the major risk factors for this company?",
                        lines=2,
                        scale=4
                    )
                    
                with gr.Row():
                    send_btn = gr.Button("📤 Send", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        def process_and_update(ticker, years):
            return analyzer.process_company_data(ticker, years)
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            
            response = analyzer.query_risk_factors(message, chat_history)

            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": response})
            
            return chat_history, ""
        
        def clear_chat():
            return []

        process_btn.click(
            fn=process_and_update,
            inputs=[ticker_input, year_checkboxes],
            outputs=[status_output],
            show_progress=True
        )
        
        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### 💡 Example Questions
                
                Try asking these questions after processing company data:
                
                - **Risk Overview**: "What are the major risk factors for this company?"
                - **Technology Risks**: "How do cybersecurity risks affect the business?"
                - **Regulatory**: "What are the regulatory compliance risks?"
                - **Competition**: "What competitive risks does the company face?"
                - **Economic**: "How might economic conditions impact the business?"
                - **Operations**: "What operational risks are mentioned?"
                - **Financial**: "What are the key financial risks?"
                
                ### 🔧 Tips
                - Select multiple years to get a comprehensive view of risk evolution
                - Ask specific questions about particular risk categories
                - Use follow-up questions to dive deeper into specific risks
                """)
    
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="127.0.0.1",  
        server_port=7860,
        share=False,  
        debug=False, 
        show_error=True
    )
