from fastapi import APIRouter, HTTPException, Depends
from app.models import QueryRequest, QueryResponse, HealthResponse, DocumentUploadResponse
from app.services.rag_pipeline import RAGPipeline
from app.dependencies import get_rag_pipeline

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Check system health"""
    return HealthResponse(
        status="healthy",
        vector_store_loaded=rag.vector_store.index.ntotal > 0,
        redis_connected=rag.cache_service.is_connected()
    )

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    rag: RAGPipeline = Depends(get_rag_pipeline)
):
    """Ask a question"""
    try:
        return rag.query(request.question, request.conversation_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/index-documents", response_model=DocumentUploadResponse)
async def index_documents(rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Index documents from data/documents directory"""
    try:
        result = rag.initialize_documents("data/documents")
        return DocumentUploadResponse(
            message="Documents indexed successfully",
            **result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/debug/store-info")
async def get_store_info(rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Get vector store information for debugging"""
    return {
        "total_vectors": rag.vector_store.index.ntotal,
        "total_documents": len(rag.vector_store.documents),
        "embedding_dimension": rag.vector_store.dimension,
        "documents_sample": [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page"),
                "content_preview": doc.page_content[:100] + "..."
            }
            for doc in rag.vector_store.documents[:3]
        ] if rag.vector_store.documents else []
    }

@router.post("/clear-index")
async def clear_index(rag: RAGPipeline = Depends(get_rag_pipeline)):
    """Clear the vector store (use before re-indexing)"""
    rag.vector_store.clear()
    rag.vector_store.save()
    return {"message": "Index cleared successfully"}