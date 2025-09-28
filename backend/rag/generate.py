# backend/rag/generate.py
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"  # inexpensive + good quality
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers using ONLY the provided context. "
    "Cite sources as (path:chunk). If the answer is not in the context, say you don't know."
)

def compose_answer(question: str, hits: List[Dict]) -> Dict:
    # Build context from retrieved hits
    blocks = []
    for h in hits:
        path = h["meta"].get("path", "unknown")
        chunk = h["meta"].get("chunk", 0)
        blocks.append(f"[{path}:{chunk}] {h['text']}")
    context = "\n\n".join(blocks) if blocks else "NO CONTEXT"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Question:\n{question}\n\nContext:\n{context}\n\n"
            "Write a concise answer (3–6 sentences) and include citations like (path:chunk)."
        }
    ]

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=250,
    )
    answer = resp.choices[0].message.content.strip()

    # Return answer + simple citation list
    citations = [
        {"path": h["meta"]["path"], "chunk": h["meta"]["chunk"], "score": h["score"]}
        for h in hits
    ]
    return {"answer": answer, "citations": citations}
