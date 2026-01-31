# Simple RAG Project

## FTL Agentic AI Training
**Lecture 12 and 13 Individual Assignment**

This project is a pure Python implementation of a Retrieval-Augmented Generation (RAG) system, utilizing **TF-IDF** and **Cosine Similarity** for document retrieval.

### Features
- **Pure Python Logic**: No heavy external libraries for core math (pure implementation of TF-IDF and Cosine Similarity).
- **Multi-Format Support**: Supports upload of `.txt`, `.pdf`, and `.docx` files.
- **Streamlit Interface**: Interactive web UI for uploading documents and testing queries.
- **Visualizations**: Displays vocabulary tables, vector breakdowns, and detailed similarity scoring.

### How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit App**:
    ```bash
    streamlit run streamlit_app.py
    ```

3.  **Upload Documents**: Use the sidebar to upload your text or PDF files.

4.  **Query**: Enter a question (e.g., "Tell me about apples") to see the most relevant documents and valid similarity scores.
