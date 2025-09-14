from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="RAG Analytics Assistant")

class Query(BaseModel):
    question: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "RAG Analytics Assistant API is running"}

@app.post("/ask")
def ask(q: Query):
    # TODO: integrate retrieval + generation
    return {
        "answer": f"Placeholder answer for: {q.question}",
        "citations": []
    }
