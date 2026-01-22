from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    conversation_id: Optional[str] = Field(None, description="For multi-turn conversations")

class SourceDocument(BaseModel):
    content: str
    source: str
    page: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    conversation_id: Optional[str] = None
    cached: bool = False

class DocumentUploadResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    redis_connected: bool