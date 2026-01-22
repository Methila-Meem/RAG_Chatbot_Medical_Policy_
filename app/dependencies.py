from functools import lru_cache
from app.services.rag_pipeline import RAGPipeline

@lru_cache()
def get_rag_pipeline():
    """Get singleton RAG pipeline instance"""
    rag = RAGPipeline()
    # Try to load existing index
    rag.vector_store.load()
    return rag