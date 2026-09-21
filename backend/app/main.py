from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.app.rag import RAGService


# ============================================================
# REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    question: str

    conversation_history: list[dict] = Field(
        default_factory=list
    )


# ============================================================
# GLOBAL RAG SERVICE
# ============================================================

rag_service = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_service

    print("=" * 70)
    print("STARTING TN SCHEME RAG API")
    print("=" * 70)

    rag_service = RAGService()

    print("=" * 70)
    print("TN SCHEME RAG API READY")
    print("=" * 70)

    yield

    print(
        "\nShutting down TN Scheme RAG API..."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Tamil Nadu Scheme RAG API",
    description=(
        "Hybrid Vector RAG + Knowledge Graph + "
        "Structured Search API for Tamil Nadu "
        "Government schemes."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Tamil Nadu Scheme RAG API is running"
        )
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "rag_loaded": rag_service is not None
    }


# ============================================================
# ASK
# ============================================================

@app.post("/ask")
def ask_scheme(
    request: AskRequest
):

    if rag_service is None:

        return {
            "error": (
                "RAG service is not initialized."
            )
        }

    result = rag_service.ask(
        question=request.question,
        conversation_history=(
            request.conversation_history
        )
    )

    return result