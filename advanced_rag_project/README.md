# Advanced RAG with Gemini

An advanced Streamlit application that performs Retrieval-Augmented Generation (RAG) using Google's Gemini models and local document indexing.

## Features
- **Document Upload**: Support for multiple text files.
- **Indexing**: TF-IDF vectorization for efficient retrieval.
- **RAG**: Combines document context with Gemini 2.0 Flash for grounded answers.
- **Transparency**: Displays source documents and relevance scores.

## Setup

1. **Environment**:
   Ensure you have the required packages installed in your environment (e.g., `ml-env`):
   ```bash
   pip install -r requirements.txt
   ```

2. **API Key**:
   You need a Google Gemini API Key. Get one from [Google AI Studio](https://aistudio.google.com/).

## How to Run

1. Navigate to the project directory:
   ```bash
   cd advanced_rag_project
   ```

2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

3. The app will open in your browser. Enter your API Key in the sidebar to start.

## Testing
Sample documents are provided in the `sample_docs/` folder.
- Upload them in the app.
- Ask questions like "What are the benefits of renewable energy?".
