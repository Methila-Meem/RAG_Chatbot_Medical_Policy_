from fastapi import FastAPI
from app.api.routes import router
from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Medical Policy RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for medical policies",
    version="1.0.0"
)

app.include_router(router, prefix="/api/v1", tags=["RAG"])

@app.get("/")
async def root():
    return {
        "message": "Medical Policy RAG Chatbot API",
        "docs": "/docs"
    }