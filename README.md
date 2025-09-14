# LLM RAG Analytics Assistant

A Retrieval-Augmented Generation (RAG) project that combines hybrid search (BM25 + embeddings) with LLM generation, citations, and A/B testing.  
Designed to showcase **production-ready AI/ML workflows** for analytics engineering and data science jobs.

---

## 🚀 Features
- **Hybrid Retrieval** → Combines dense embeddings + sparse BM25 for accurate search.  
- **Grounded Generation** → Answers always include **citations** to source documents.  
- **A/B Testing** → Compare prompt templates, retrieval configs, or rerankers.  
- **Guardrails** → Schema validation, PII filtering, and confidence thresholds.  
- **Evaluation** → Offline with RAGAS metrics (correctness, faithfulness, recall).  
- **Observability** → Latency, token usage, retrieval hit-rate, and user feedback tracking.  
- **UI Demo** → Streamlit/React interface for end-to-end testing.  

---

## 🏗️ Tech Stack
- **Backend**: Python, FastAPI  
- **RAG**: LangChain / LlamaIndex  
- **Embeddings**: OpenAI / BGE  
- **Vector DB**: FAISS (local) / Milvus / Pinecone  
- **UI**: Streamlit (easy demo) or React (production-style)  
- **Evaluation**: RAGAS, custom notebooks  
- **Infra**: Docker, GitHub Actions  

---

## 📂 Repo Structure
