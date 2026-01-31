import streamlit as st
import pandas as pd
import os
from rag_core import build_vocabulary, compute_idf, text_to_vector, cosine_similarity, get_magnitude, tokenize
from file_utils import read_file_content

st.set_page_config(page_title="Simple RAG Engine", page_icon="🔍", layout="wide")

st.title("🔍 Simple RAG Engine")
st.markdown("A pure Python implementation of Retrieval-Augmented Generation using **TF-IDF** and **Cosine Similarity**.")

# --- Sidebar: Upload ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Select .txt, .pdf, or .docx files to build your knowledge base."
    )

    if not uploaded_files:
        st.info("👆 Please upload documents to begin.")

# --- Main Logic ---

if uploaded_files:
    documents = {}
    for uploaded_file in uploaded_files:
        
        st.write(f"**Vocabulary Size:** {len(vocabulary)} words")
        st.write(f"**Documents:** {len(documents)}")
        
        # Show Vocabulary Table
        vocab_df = pd.DataFrame({
            "Term": vocabulary,
            "IDF Score": [idf_scores[t] for t in vocabulary]
        })
        st.dataframe(vocab_df, height=200, use_container_width=True)

        st.subheader("2. Document Vectors (TF-IDF)")
        # Calculate vectors
        doc_vectors = []
        for content in doc_contents:
            doc_vectors.append(text_to_vector(content, vocabulary, idf_scores))
        
        # Show Vectors Table
        vectors_df = pd.DataFrame(doc_vectors, columns=vocabulary, index=doc_names)
        st.dataframe(vectors_df, height=300, use_container_width=True)

    # --- Step 2: Query ---
    st.divider()
    st.header("🤖 Query System")
    
    query = st.text_input("Enter your query:", placeholder="e.g., Tell me about apples")
    
    if query:
        # Re-compute these if not in expander, or just use variables from above (Python scoping allows it in Streamlit script)
        if 'vocabulary' not in locals():
            vocabulary = build_vocabulary(doc_contents)
            idf_scores = compute_idf(doc_contents, vocabulary)
            doc_vectors = []
            for content in doc_contents:
                doc_vectors.append(text_to_vector(content, vocabulary, idf_scores))
        
        # Vectorize Query
        query_vector = text_to_vector(query, vocabulary, idf_scores)
        query_mag = get_magnitude(query_vector)
        
        # Display Query Vector
        st.subheader("Step 2: Query Vectorization")
        
        # Filter non-zero terms for display
        non_zero_query = {vocabulary[i]: val for i, val in enumerate(query_vector) if val > 0}
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**Processed Query Tokens:**")
            st.code(tokenize(query))
        with col2:
            st.write("**Non-Zero Vector Dimensions:**")
            if non_zero_query:
                st.write(non_zero_query)
            else:
                st.warning("No vocabulary words found in query!")

        # --- Step 3: Similarity & Ranking ---
        st.subheader("Step 3: Cosine Similarity & Ranking")
        
        results = []
        for i, doc_vec in enumerate(doc_vectors):
            score = cosine_similarity(query_vector, doc_vec)
            doc_mag = get_magnitude(doc_vec)
            dot_product = sum(query_vector[j] * doc_vec[j] for j in range(len(query_vector)))
            
            results.append({
                "Document": doc_names[i],
                "Score": score,
                "Dot Product": dot_product,
                "Doc Magnitude": doc_mag,
                "Query Magnitude": query_mag,
                "Content": doc_contents[i]
            })
            
        # Sort results
        results.sort(key=lambda x: x["Score"], reverse=True)
        
        # Display Top Result prominently
        top_result = results[0]
        
        st.success(f"🏆 Top Result: **{top_result['Document']}** (Similarity: {top_result['Score']:.4f})")
        
        with st.container(border=True):
            st.markdown(f"### {top_result['Document']}")
            st.text(top_result['Content'])
        
        # Detailed Calculation Table
        st.markdown("#### Similarity Breakdown")
        results_df = pd.DataFrame(results).drop(columns=["Content"])
        st.dataframe(
            results_df.style.background_gradient(subset=["Score"], cmap="Greens").format("{:.4f}", subset=["Score", "Dot Product", "Doc Magnitude", "Query Magnitude"]), 
            use_container_width=True
        )

else:
    st.write("waiting for upload...")
