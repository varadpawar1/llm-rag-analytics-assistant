# backend/rag/ingest.py
import os
from pathlib import Path
import pickle
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = Path("data/raw_docs")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
DIM = 1536  # embedding dim

def read_docs():
    docs = []
    for p in DATA_DIR.glob("*.txt"):
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"text": text, "path": str(p)})
    return docs

def embed_texts(texts):
    # Batch to be safe
    B = 64
    vecs = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
    arr = np.array(vecs, dtype="float32")
    # L2 normalize so dot product ≈ cosine
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr

def build_index():
    docs = read_docs()
    if not docs:
        print("No docs found in data/raw_docs. Add .txt files and rerun.")
        return
    texts = [d["text"] for d in docs]
    X = embed_texts(texts)  # (N, DIM), normalized

    # Save matrix + metadata
    np.save(INDEX_DIR / "embeddings.npy", X)
    with open(INDEX_DIR / "metadatas.pkl", "wb") as f:
        pickle.dump(docs, f)

    print(f"Indexed {len(docs)} docs into {INDEX_DIR}")

if __name__ == "__main__":
    build_index()
