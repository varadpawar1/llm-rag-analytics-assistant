# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag.embeddings import embed_query
from backend.rag.retriever import search
from backend.rag.generate import compose_answer

app = FastAPI(title="RAG Analytics Assistant")

class Query(BaseModel):
    question: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "RAG Analytics Assistant API is running"}

@app.post("/ask")
def ask(q: Query):
    # 1) embed the question
    qvec = embed_query(q.question)
    # 2) retrieve top-k relevant docs/snippets
    hits = search(qvec, top_k=q.top_k)
    # 3) generate a concise answer grounded in context
    gen = compose_answer(q.question, hits)
    return {
        "answer": gen["answer"],
        "citations": gen["citations"],
        "snippets": [h["text"] for h in hits]  # keep for debugging/demo
    }
