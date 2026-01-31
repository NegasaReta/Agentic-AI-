import streamlit as st
import google.generativeai as genai
from google.api_core import exceptions
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Configure page settings
st.set_page_config(
    page_title="Advanced RAG with Gemini",
    page_icon="🧠",
    layout="wide"
)

def generate_with_retry(model, prompt, retries=3, delay=5):
    """
    Helper function to generate content with retry logic for rate limits.
    """
    for attempt in range(retries):
        try:
            return model.generate_content(prompt)
        except exceptions.ResourceExhausted as e:
            if attempt < retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
            else:
                raise e
        except Exception as e:
            raise e

def main():
    st.title("🧠 Advanced RAG System with Gemini")
    st.markdown("Upload documents, ask questions, and get grounded answers using Gemini.")

    # Application state initialization
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "vectorizer" not in st.session_state:
        st.session_state.vectorizer = None
    if "api_configured" not in st.session_state:
        st.session_state.api_configured = False

    # Sidebar Configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Gemini API Key", type="password")
        model_name = st.selectbox(
            "Select Model",
            ["gemini-3.0-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        )
        
        # Validation
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Simple call to verify key works (list models or similar, but configure is usually lazy)
                # We can try to list models to verify auth
                genai.list_models()
                st.session_state.api_configured = True
                st.success("API Connection Validated! ✅")
            except Exception as e:
                st.session_state.api_configured = False
                st.error(f"Invalid API Key: {e}")
        else:
            st.session_state.api_configured = False
            st.warning("Please enter your Gemini API Key to proceed.")

    # Main Area Logic
    if st.session_state.api_configured:
        st.info(f"Connected to **{model_name}**. Ready to process documents.")
        
        # Step 2: Upload Documents
        uploaded_files = st.file_uploader(
            "Upload Text Documents", 
            type=['txt', 'md'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                documents = []
                doc_names = []
                
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Robust reading handling
                    try:
                        string_data = uploaded_file.getvalue().decode("utf-8")
                        documents.append(string_data)
                        doc_names.append(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                if documents:
                    st.session_state.documents = documents
                    st.session_state.doc_names = doc_names
                    st.success(f"✓ Loaded {len(documents)} documents successfully!")
                    
                    # Step 3: Document Indexing
                    with st.spinner("Indexing documents..."):
                        try:
                            # 1. Create Vectorizer
                            vectorizer = TfidfVectorizer(stop_words='english')
                            
                            # 2. Fit and Transform documents
                            vectors = vectorizer.fit_transform(documents)
                            
                            # 3. Store in session state
                            st.session_state.embeddings = vectors
                            st.session_state.vectorizer = vectorizer
                            
                            st.info("✓ Indexing complete! Vectors stored in memory.")
                            
                        except Exception as e:
                            st.error(f"Error during indexing: {e}")
                else:
                    st.warning("No valid text extracted from uploaded files.")
        
        st.markdown("---")
        
        # Step 4: User Enters Query
        if st.session_state.vectorizer:
            query = st.text_input("Ask a question about your documents:")
            
            if query:
                # Step 5: Retrieve Relevant Documents
                with st.spinner("Retrieving relevant context..."):
                    # Transform query
                    query_vector = st.session_state.vectorizer.transform([query])
                    
                    # Calculate similarity
                    similarities = cosine_similarity(query_vector, st.session_state.embeddings).flatten()
                    
                    # Get top 3 indices
                    top_indices = similarities.argsort()[:-4:-1] # Top 3
                    
                    # Extract context
                    context = ""
                    sources = []
                    for idx in top_indices:
                        if similarities[idx] > 0.1: # relevance threshold
                            doc_content = st.session_state.documents[idx]
                            doc_name = st.session_state.doc_names[idx]
                            
                            # Taking a snippet (first 1000 chars or reasonable chunk around match? 
                            # TF-IDF gives doc-level score, so we take the whole doc or a large chunk)
                            # For simplicity/correctness with TF-IDF doc-level retrieval, we feed the doc content.
                            # To save tokens, we might truncate. Let's truncate to 2000 chars per doc.
                            snippet = doc_content[:2000] 
                            context += f"\nDocument: {doc_name}\nContent: {snippet}\nRelevance: {similarities[idx]:.2f}\n"
                            
                            sources.append((doc_name, similarities[idx]))
                    
                    st.write("### Retrieved Context")
                    with st.expander("Show Retrieved Documents"):
                        for src, score in sources:
                            st.write(f"- {src} (Score: {score:.2f})")
                            
                    # Step 6: Send to LLM with Context
                    if context:
                        with st.spinner("Generating answer..."):
                             system_prompt = f"""
                            You are a helpful assistant. Answer the user's question based ONLY on the following documents.
                            If the answer is not in the documents, say "I cannot find the answer in the provided documents."
                            
                            Documents:
                            {context}
                            
                            User Query: {query}
                            
                            Answer:
                            """
                             
                             try:
                                 model = genai.GenerativeModel(model_name)
                                 # Use retry logic
                                 response = generate_with_retry(model, system_prompt)
                                 
                                 # Step 7: Display Response
                                 st.markdown("### 🤖 Gemini Response")
                                 st.write(response.text)
                                 
                                 # Token Usage (if available)
                                 if response.usage_metadata:
                                     st.markdown("---")
                                     col1, col2, col3 = st.columns(3)
                                     col1.metric("Prompt Tokens", response.usage_metadata.prompt_token_count)
                                     col2.metric("Response Tokens", response.usage_metadata.candidates_token_count)
                                     col3.metric("Total Tokens", response.usage_metadata.total_token_count)
                             
                             except exceptions.ResourceExhausted:
                                 st.error("⚠️ API Quota Exceeded. The free tier limit for Gemini has been reached. Please try again in 60 seconds or switch to a paid plan.")
                             except Exception as e:
                                 st.error(f"Error calling Gemini API: {e}")
                    else:
                        st.warning("No relevant documents found matching your query.")
    else:
        st.markdown("---")
        st.markdown("### 👈 Connect to Gemini in the sidebar to start!")

if __name__ == "__main__":
    main()
