# Financial Document Analyzer

A web-based tool to interactively extract and analyze risk factors from a company's Form 10-K filings. Input a stock ticker and year to chat with AI about the company's major risks directly from SEC filings, focusing on up-to-date risk disclosures.

## Features

- Select any public company by its stock ticker.
- Choose a year (2006–2025) to retrieve Form 10-K risk factors.
- Conversational chat interface for asking in-depth questions about company risks.
- AI-powered summarization and breakdown of risk factors.
- Visual status showing processed company and year selection.

## How It Works

1. **Select Stock Ticker and Year**: Use the interface to pick the company ticker (e.g., AAPL) and year of analysis.

   ![Stock and Year Selection](brave_screenshot.jpg)

2. **Process Company Data**: The app parses the selected company's risk factors from its 10-K report for the chosen year.

   ![Status and Processing](brave_screenshot-1.jpg)

3. **Chat with AI**: Ask any questions about major risk factors. The AI gives summarized and detailed views, such as competition, economic conditions, and reputation impacts.

## Example Use Case

- **User:** What are the major risk factors for this company?
- **AI:** Provides summary and breakdown (e.g., competition, economic conditions, reputation, etc.).

## Technologies Used

- Python
- LangChain
- Gemini API
- Financial document scraping (SEC 10-Ks)
- Conversational UI (Gradio, Streamlit, or custom frontend)

## Credits

- Developed by Vikas1177  
