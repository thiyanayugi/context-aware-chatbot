from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

summary = """Here is a summary of the key points from the conversation:

- The user's name is Yugi and they love using the FastAPI Python web framework
- FastAPI is known for being fast, efficient, and easy to use, with automatic documentation generation and built-in data validation
- FastAPI also has strong support for asynchronous programming
- The assistant asked if Yugi has any specific projects or use cases in mind where they plan to utilize FastAPI, and offered to provide further guidance based on Yugi's needs"""

queries = [
    "What do I love",
    "What is my favorite framework?",
    "Does Yugi like FastAPI?",
    "backend development preferences"
]

# Encode
summary_emb = model.encode(summary)
query_embs = model.encode(queries)

# Compute cosine similarity
scores = util.cos_sim(query_embs, summary_emb)

for q, s in zip(queries, scores):
    print(f"Query: '{q}' -> Score: {s.item()}")
