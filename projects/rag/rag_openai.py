# rag.py
# Minimal Retrieval-Augmented Generation in ONE Python file

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# -----------------------------
# 1. Configuration
# -----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"   # change if needed
TOP_K = 3

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 2. Knowledge Base (documents)
# -----------------------------
documents = [
    "RAG combines retrieval with text generation to ground LLM outputs.",
    "FAISS is a library for efficient similarity search over vectors.",
    "Embeddings map text into dense vector space where similarity is distance.",
    "Large Language Models can hallucinate without external grounding."
]

# -----------------------------
# 3. Embed & Index
# -----------------------------
embedder = SentenceTransformer(EMBED_MODEL)

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# -----------------------------
# 4. Retrieve
# -----------------------------
def retrieve(query, k=TOP_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    return [documents[i] for i in indices[0]]

# -----------------------------
# 5. Generate
# -----------------------------
def rag_answer(query):
    context = retrieve(query)

    prompt = f"""
You are answering using retrieved context only.

Context:
{chr(10).join(context)}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content

# -----------------------------
# 6. Run
# -----------------------------
if __name__ == "__main__":
    question = "Why does RAG reduce hallucinations?"
    answer = rag_answer(question)
    print("\nAnswer:\n", answer)
