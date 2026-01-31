
import math
import re
from collections import Counter

def tokenize(text):
    """
    Simple tokenizer that converts text to lowercase, removes punctuation,
    and splits into words.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Split by whitespace
    words = text.split()
    # Simple stemming logic
    stemmed_words = []
    for word in words:
        # 1. 'ies' -> 'y' (e.g. companies -> company)
        if word.endswith('ies') and len(word) > 3:
            stemmed_words.append(word[:-3] + 'y')
        # 2. 'es' -> '' only for specific endings (boxes -> box, but apples -> apple via 's' rule)
        # This is a simplification. 'apples' ends in 'es' but we want to strip 's' to get 'apple'.
        # We'll skip strict 'es' handling for simplicity and just try to handle 's'.
        
        # 3. 's' -> '' (e.g. apples -> apple, cats -> cat), but not 'ss' matches
        elif word.endswith('s') and not word.endswith('ss') and len(word) > 2:
            stemmed_words.append(word[:-1])
        else:
            stemmed_words.append(word)
    return stemmed_words

def build_vocabulary(documents):
    """
    Builds a sorted list of unique words from a list of documents.
    """
    unique_words = set()
    for doc in documents:
        words = tokenize(doc)
        unique_words.update(words)
    return sorted(list(unique_words))

def compute_idf(documents, vocabulary):
    """
    Computes Inverse Document Frequency (IDF) for each word in the vocabulary.
    IDF(t) = log(N / df(t))
    """
    N = len(documents)
    idf_values = {}
    
    for word in vocabulary:
        # Count number of documents containing the word
        doc_count = 0
        for doc in documents:
            if word in tokenize(doc):
                doc_count += 1
        
        # Avoid division by zero (shouldn't happen if vocab comes from docs)
        if doc_count > 0:
            idf_values[word] = math.log(N / doc_count)
        else:
            idf_values[word] = 0.0
            
    return idf_values

def text_to_vector(text, vocabulary, idf_scores=None):
    """
    Converts text to a TF-IDF vector based on the vocabulary.
    If idf_scores is None, returns just TF vector.
    """
    words = tokenize(text)
    word_counts = Counter(words)
    total_words = len(words)
    
    vector = []
    for term in vocabulary:
        # TF = (count of term in doc) / (total words in doc)
        # However, for simple RAG, sometimes raw count is used, but standard TF is normalized.
        tf = word_counts[term] / total_words if total_words > 0 else 0
        
        if idf_scores:
            idf = idf_scores.get(term, 0)
            vector.append(tf * idf)
        else:
            vector.append(tf)
            
    return vector

def get_magnitude(vec):
    """Calculates the magnitude (Euclidean norm) of a vector."""
    return math.sqrt(sum(x * x for x in vec))

def cosine_similarity(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.
    Similarity = (A . B) / (||A|| * ||B||)
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of same length")
        
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    magnitude1 = get_magnitude(vec1)
    magnitude2 = get_magnitude(vec2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
        
    return dot_product / (magnitude1 * magnitude2)
