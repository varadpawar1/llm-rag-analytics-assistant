# backend/rag/retriever.py
from pathlib import Path
import pickle
import numpy as np
from typing import List, Dict

INDEX_DIR = Path("data/index")
EMBED_FILE = INDEX_DIR / "embeddings.npy"
META_FILE = INDEX_DIR / "metadatas.pkl"

def _load():
    X = np.load(EMBED_FILE)  # (N, 1536) normalized embeddings
    with open(META_FILE, "rb") as f:
        docs = pickle.load(f)  # list of {"text":..., "path":...}
    return X, docs

def search(query_vec: np.ndarray, top_k: int = 5) -> List[Dict]:
    # normalize query
    q = query_vec.astype("float32")
    q = q / (np.linalg.norm(q) + 1e-12)

    X, docs = _load()
    # cosine similarity = dot since vectors are normalized
    sims = X @ q  # (N,)
    idx = np.argsort(-sims)[:top_k]  # top-k highest similarity

    hits = []
    for i in idx:
        hits.append({
            "score": float(sims[i]),
            "text": docs[i]["text"],
            "meta": {"path": docs[i]["path"], "chunk": 0}
        })
    return hits
