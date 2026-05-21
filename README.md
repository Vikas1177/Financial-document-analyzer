#  Financial Document Analyzer

A sophisticated AI-powered web application that intelligently analyzes SEC Form 10-K filings, enabling investors and financial analysts to extract, understand, and ask detailed questions about company risk factors. 

---

##  Overview

This project leverages **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** to provide intelligent analysis of financial documents. Users can select any publicly traded company by its stock ticker and specific years, and then have a natural conversation with an AI assistant about the company's risk factors discovered in SEC filings.



## Step-by-Step Processing Flow:

1. **Input Stage**: User enters stock ticker (e.g., AAPL) and selects desired years
2. **Retrieval Stage**: Application fetches 10-K document URLs from SEC EDGAR
3. **Extraction Stage**: Risk factors are extracted from specified filings
4. **Processing Stage**: Documents are split into chunks and converted to embeddings
5. **Indexing Stage**: Vector embeddings are stored in Chroma database
6. **Query Stage**: User questions are converted to embeddings
7. **Retrieval Stage**: Semantically similar chunks are retrieved
8. **Generation Stage**: LLM generates informed answers based on retrieved context
9. **Response Stage**: Answer is presented to user in conversational format

---

##  Technology Stack

### **Core AI/ML Technologies**

| Technology | Purpose | Version |
|-----------|---------|---------|
| **LangChain** | Framework for building LLM applications | 0.3.27 |
| **LangChain Community** | Extended integrations | 0.3.27 |
| **LangChain Text Splitters** | Intelligent document chunking | 0.3.11 |
| **Chroma** | Vector database for embeddings | 1.5.9 |
| **Google Generative AI (Gemini)** | Advanced LLM for answer generation | 2.0 Flash |
| **HuggingFace Embeddings** | Text-to-vector conversion (all-MiniLM-L6-v2) | Full integration |

### **Data Source & Processing**

| Technology | Purpose | Description |
|-----------|---------|-------------|
| **SEC API** | Financial document source | 1.0.32 - Accesses SEC EDGAR database |
| **Requests** | HTTP client | 2.32.3 - Fetches SEC data |

### **Web Interface & Deployment**

| Technology | Purpose | Version |
|-----------|---------|---------|
| **Gradio** | Web UI framework | 5.40.0 |
| **Python** | Core language | 3.x |
| **Python-dotenv** | Environment configuration | 1.0.1 |

### **Architecture Patterns**

- **RAG (Retrieval-Augmented Generation)**: Combines document retrieval with LLM generation
- **Vector Embeddings**: Semantic understanding through embeddings
- **Prompt Engineering**: Structured prompts for consistent LLM responses
- **LangChain Chains**: Composable LLM application components

---

##  Dependencies & Requirements

All dependencies are listed in `requirements.txt`:


---

##  Key Features

### 1. **Multi-Year Analysis**
   - Select and analyze risk factors from up to 20 years of historical data
   - Track how risk profiles evolve over time
   - Compare risk factors across different years

### 2. **Intelligent Document Extraction**
   - Automatically extracts Section 1A (Risk Factors) from 10-K filings
   - Processes official SEC documents from EDGAR database
   - Handles multiple filings seamlessly

### 3. **Semantic Search & Retrieval**
   - Converts documents to semantic embeddings
   - Performs similarity-based search for relevant content
   - Returns most contextually relevant sections for each query

### 4. **Conversational AI Interface**
   - Natural language questions about risk factors
   - Context-aware responses from LLM
   - Follow-up questioning capability
   - Maintains conversation history

### 5. **Comprehensive Risk Analysis**
   - Competition risks
   - Economic & market risks
   - Regulatory & compliance risks
   - Technology & cybersecurity risks
   - Operational risks
   - Financial risks
   - Reputational risks

### 6. **User-Friendly Web Interface**
   - Intuitive Gradio-based UI
   - Real-time processing status
   - Example questions provided
   - Copy-to-clipboard functionality

---

##  Benefits & Use Cases

### **For Individual Investors**
-  **Informed Investment Decisions**: Understand key risks before investing
-  **Time Savings**: Analyze lengthy 10-K documents in minutes instead of hours
-  **Risk Assessment**: Identify potential threats to company performance
-  **Deep Insights**: Ask specific questions about risk factors

### **For Financial Analysts**
-  **Comprehensive Analysis**: Multi-year risk evolution tracking
-  **Faster Research**: AI-accelerated document analysis
-  **Due Diligence**: Streamlined risk factor review for M&A activity
-  **Comparative Analysis**: Compare risk profiles across companies

### **For Risk Managers**
-  **Risk Identification**: Uncover hidden or non-obvious risks
-  **Compliance Monitoring**: Track regulatory risk changes
-  **Trend Analysis**: Monitor how risks change year-to-year
-  **Document Management**: Organized risk factor reference

### **For Business Analysts**
-  **Competitive Intelligence**: Understand competitor risk profiles
-  **Market Analysis**: Identify industry-wide risk trends
-  **Strategic Planning**: Inform business strategy with risk insights
-  **Report Generation**: Extract key risk factors for reports

### **Time & Cost Efficiency**
-  **Speed**: Analyze documents in minutes vs. hours
-  **Cost Reduction**: Reduce manual document review labor
-  **Scalability**: Analyze multiple companies simultaneously
-  **Automation**: Fully automated extraction and analysis pipeline

### **Data-Driven Benefits**
-  **Structured Insights**: Convert unstructured text to actionable data
-  **Semantic Understanding**: AI understands context and meaning
-  **Multiple Perspectives**: Ask same question different ways
-  **Pattern Recognition**: Identify risk patterns and trends

---

##  Getting Started

### Prerequisites
- Python 3.8 or higher
- Google API Key (for Gemini LLM access)
- SEC API Key (optional, for enhanced rate limits)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Vikas1177/Financial-document-analyzer.git
   cd Financial-document-analyzer
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: v
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables: Create a .env file in the root directory**:
   GOOGLE_API_KEY=your_google_api_key_here
   SEC_API_KEY=your_sec_api_key_here
5. **Run the application**:
   ```bash
   python -m src.gradio_app
   ```

## Model Parameters
Embedding Model: all-MiniLM-L6-v2 (384-dimensional vectors)
LLM Model: gemini-2.0-flash (Fast, accurate responses)
Temperature: 0.1 (Deterministic, factual responses)
Chunk Size: 1000 tokens
Chunk Overlap: 200 tokens
Retrieval K: 5 (top 5 most relevant chunks)

## How It Works: Technical Details
1. **SEC Data Retrieval**
   Uses company ticker to look up CIK (Central Index Key)
   Queries SEC EDGAR API for all 10-K filings
   Returns URLs for specified years
2. **Document Extraction**
   Extracts Section 1A (Risk Factors) from HTML/plain text 10-K documents
   Preserves original text structure and content
   Handles multiple filings in batch
3. **Text Processing**
   Splits documents using RecursiveCharacterTextSplitter
   Custom separators: filing boundaries, paragraphs, sentences, words
   Creates overlapping chunks for context preservation
4. **Vector Embeddings**
   Converts each chunk to 384-dimensional vector
   Uses HuggingFace's efficient embedding model
   Stores in Chroma vector database with metadata
5. **Semantic Retrieval**
   User query converted to embedding
   Cosine similarity search finds relevant chunks
   Returns top-K most similar documents
6. **LLM Generation**
   Retrieves documents provide context
   Gemini LLM generates comprehensive answer
   Prompt engineering ensures factual, sourced responses


## Credits

- Developed by Vikas1177  
