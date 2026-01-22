from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.config import get_settings

settings = get_settings()

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple documents"""
        return self.model.encode(texts, show_progress_bar=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.model.encode([query])[0]