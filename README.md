# 🇮🇳 Tamil Nadu Government Scheme RAG

An AI-powered Retrieval-Augmented Generation (RAG) application for searching and understanding Tamil Nadu Government schemes.

The application combines:

- Web-scraped Tamil Nadu Government scheme data
- FAISS vector search
- Knowledge Graph retrieval
- Structured JSON filtering
- LLM-based query understanding
- Hybrid retrieval
- OpenAI GPT-4o-mini
- FastAPI backend
- Streamlit frontend
- Conversation-aware follow-up questions

---

## 📌 Project Overview

The goal of this project is to build an intelligent assistant that allows users to ask natural-language questions about Tamil Nadu Government schemes.

Instead of relying on a simple keyword search, the application combines multiple retrieval techniques to identify relevant schemes.

Example questions:

_Which schemes provide subsidies to farmers?

Which schemes provide grants?

Which agriculture schemes support pulses?

Which schemes are available for farmers in Chennai?

Which schemes are sponsored by the State?

What about pulses?_

**Architecture**

                 Tamil Nadu Government Website
                              │
                              ▼
                        Web Scraping
                              │
                              ▼
                       schemes.json
                              │
                 ┌────────────┼─────────────┐
                 │            │             │
                 ▼            ▼             ▼
              FAISS       Knowledge      Structured
             Vector DB       Graph          JSON
                 │            │             │
                 └────────────┼─────────────┘
                              ▼
                     Hybrid Retrieval
                              │
                              ▼
                     Query Understanding
                              │
                              ▼
                        GPT-4o-mini
                              │
                              ▼
                          FastAPI
                              │
                              ▼
                         Streamlit
                              │
                              ▼
                           User

      **Project Structure**

      W3D1 - TN Scheme RAG/
│
├── backend/
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       └── rag.py
│
├── frontend/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── schemes.json
│   │   └── llm_knowledge_graph.gpickle
│   │
│   └── processed/
│
├── scripts/
│   ├── create_knowledge_graph.py
│   ├── test_llm_graph_extraction.py
│   └── query_understanding.py
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── .env
├── .gitignore
├── README.md
└── venv/
                 

