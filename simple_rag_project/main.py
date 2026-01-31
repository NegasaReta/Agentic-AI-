
import os
import glob
import sys
from rag_core import build_vocabulary, compute_idf, text_to_vector, cosine_similarity

def load_documents(directory):
    """Loads all text files from the given directory."""
    documents = {}
    filepaths = glob.glob(os.path.join(directory, "*.txt"))
    
    if not filepaths:
        print(f"No .txt files found in {directory}")
        return {}
        
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                documents[os.path.basename(filepath)] = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return documents

def main():
    print("=== Simple Pure Python RAG System ===")
    
    # 1. Load Documents
    print("Step 1: Upload Documents")
    print("Enter the paths to your text files (comma-separated), or press Enter to use default 'sample_docs/':")
    user_paths = input("> ").strip()
    
    docs_map = {}
    
    if user_paths:
        paths = [p.strip() for p in user_paths.split(',')]
        for valid_path in paths:
            # Remove quotes if user added them
            valid_path = valid_path.strip('"').strip("'")
            if os.path.isfile(valid_path):
                try:
                    with open(valid_path, 'r', encoding='utf-8') as f:
                        docs_map[os.path.basename(valid_path)] = f.read()
                except Exception as e:
                    print(f"Error reading {valid_path}: {e}")
            else:
                print(f"Warning: File not found: {valid_path}")
    else:
        # Default to sample_docs
        doc_dir = "sample_docs"
        if not os.path.exists(doc_dir):
            doc_dir = "."
        print(f"Loading documents from '{doc_dir}'...")
        docs_map = load_documents(doc_dir)
    
    if not docs_map:
        print("No documents loaded. Exiting.")
        return

    doc_names = list(docs_map.keys())
    doc_contents = list(docs_map.values())
    
    print(f"Successfully uploaded {len(docs_map)} documents: {', '.join(doc_names)}")
    print("-" * 30)
    
    # 2. Build Vocabulary and IDF
    print("Building vocabulary...")
    vocabulary = build_vocabulary(doc_contents)
    print(f"Vocabulary size: {len(vocabulary)} words")
    
    print("Computing IDF scores...")
    idf_scores = compute_idf(doc_contents, vocabulary)
    
    # 3. Vectorize Documents
    print("Vectorizing documents...")
    doc_vectors = []
    for content in doc_contents:
        vec = text_to_vector(content, vocabulary, idf_scores)
        doc_vectors.append(vec)
        
    # 4. User Query Loop
    print("\nSystem ready! Type 'exit' to quit.")
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
        except EOFError:
            break
            
        if query.lower() in ('exit', 'quit'):
            break
            
        if not query:
            continue
            
        # Vectorize query
        # Note: We use the same vocabulary and IDF scores derived from docs
        query_vector = text_to_vector(query, vocabulary, idf_scores)
        
        # [DEMO] Show non-zero query terms to prove vectorization
        non_zero_query = {vocabulary[i]: round(val, 4) for i, val in enumerate(query_vector) if val > 0}
        print(f"\n[STEP 2] Converted Query to Vector (showing non-zero dimensions):\n{non_zero_query}")
        
        # Calculate similarities
        results = []
        print("\n[STEP 3] Computing Cosine Similarity with Documents:")
        for i, doc_vec in enumerate(doc_vectors):
            score = cosine_similarity(query_vector, doc_vec)
            print(f"  - Similarity vs {doc_names[i]}: {score:.4f}")
            results.append((doc_names[i], score, doc_contents[i]))
            
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Display Results
        print("\n--- Top Results ---")
        for rank, (name, score, content) in enumerate(results[:3], start=1):
            print(f"{rank}. {name} (Similarity: {score:.4f})")
            if rank == 1:
                print(f"   Full Content:\n\"{content.strip()}\"")
            else:
                snippet = content[:100].replace('\n', ' ') + "..." if len(content) > 100 else content
                print(f"   Content: \"{snippet.strip()}\"")
            print("-" * 30)

if __name__ == "__main__":
    main()
