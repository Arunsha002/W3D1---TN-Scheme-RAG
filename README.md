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

```text  
Which schemes provide subsidies to farmers?
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
Which schemes provide grants?

Which agriculture schemes support pulses?

Which schemes are available for farmers in Chennai?

Which schemes are sponsored by the State?

What about pulses?


