# rag_gemini.py
# Minimal Retrieval-Augmented Generation using Google Gemini
# Requires: pip install google-generativeai

import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -----------------------------
# 1. Configuration
# -----------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash" 
TOP_K = 3

load_dotenv()

# Configure Gemini
# Get your key from: https://aistudio.google.com/app/apikey
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    # Fallback to asking user input if not in env
    print("Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
    # In a real app you might raise an error, but here we'll let the library handle the missing key error or prompt.
else:
    genai.configure(api_key=api_key)

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
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)

print("Embedding documents...")
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
# 5. Generate with Gemini
# -----------------------------
def rag_answer_gemini(query):
    context = retrieve(query)
    
    # Construct the prompt
    prompt = f"""
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
{chr(10).join(context)}

Question:
{query}

Answer:
"""
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {e}"

# -----------------------------
# 6. Run
# -----------------------------
if __name__ == "__main__":
    question = "Why does RAG reduce hallucinations?"
    print(f"\nQuestion: {question}")
    
    if not api_key:
        print("\n[!] Please set GEMINI_API_KEY in your .env file or environment variables to run the generation step.")
        print("You can get a free key here: https://aistudio.google.com/app/apikey")
    else:
        answer = rag_answer_gemini(question)
        print("\nAnswer:\n", answer)
