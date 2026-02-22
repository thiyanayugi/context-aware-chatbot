"""
Semantic similarity scoring utility.

This script demonstrates how conversation summaries are scored against user queries
using sentence embeddings and cosine similarity. It's used for testing and validating
the relevance scoring mechanism used in the chatbot's memory retrieval system.
"""

from sentence_transformers import SentenceTransformer, util
from typing import List


def compute_relevance_scores(
    summary: str,
    queries: List[str],
    model_name: str = 'all-MiniLM-L6-v2'
) -> List[tuple[str, float]]:
    """
    Compute semantic relevance scores between a summary and multiple queries.
    
    Args:
        summary: The conversation summary to score against
        queries: List of user queries to evaluate
        model_name: Name of the sentence transformer model to use
        
    Returns:
        List of (query, score) tuples with cosine similarity scores
    """
    # Load pre-trained sentence transformer model
    model = SentenceTransformer(model_name)
    
    # Encode summary and queries into dense vector representations
    summary_embedding = model.encode(summary)
    query_embeddings = model.encode(queries)
    
    # Compute cosine similarity between each query and the summary
    scores = util.cos_sim(query_embeddings, summary_embedding)
    
    # Return paired results
    return [(query, score.item()) for query, score in zip(queries, scores)]


if __name__ == "__main__":
    # Example conversation summary
    summary = """Here is a summary of the key points from the conversation:

- The user's name is Yugi and they love using the FastAPI Python web framework
- FastAPI is known for being fast, efficient, and easy to use, with automatic documentation generation and built-in data validation
- FastAPI also has strong support for asynchronous programming
- The assistant asked if Yugi has any specific projects or use cases in mind where they plan to utilize FastAPI, and offered to provide further guidance based on Yugi's needs"""

    # Test queries to evaluate relevance
    test_queries = [
        "What do I love",
        "What is my favorite framework?",
        "Does Yugi like FastAPI?",
        "backend development preferences"
    ]
    
    # Compute and display results
    results = compute_relevance_scores(summary, test_queries)
    
    print("Semantic Similarity Scores:")
    print("-" * 60)
    for query, score in results:
        print(f"Query: '{query}'")
        print(f"  → Relevance Score: {score:.4f}")
        print()
