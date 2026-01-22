from typing import List, Dict, Optional
from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.services.llm_service import LLMService
from app.services.cache_service import CacheService
from app.models import QueryResponse, SourceDocument
from app.config import get_settings
import uuid

settings = get_settings()

class RAGPipeline:
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.embedding_service = EmbeddingService()
        self.vector_store = FAISSVectorStore()
        self.llm_service = LLMService()
        self.cache_service = CacheService()
        self.conversation_store: Dict[str, List[dict]] = {}
    
    def initialize_documents(self, directory: str) -> Dict:
        """Load and index documents"""
        print(f"Loading documents from: {directory}")
        
        # Load documents
        documents = self.document_loader.load_documents_from_directory(directory)
        print(f"Loaded {len(documents)} documents")
        
        if not documents:
            raise Exception("No documents found in directory")
        
        # Split into chunks
        chunks = self.document_loader.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        if not chunks:
            raise Exception("No chunks created from documents")
        
        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Add to vector store
        self.vector_store.add_documents(chunks, embeddings)
        print(f"Added {len(chunks)} documents to vector store")
        
        # Save index
        self.vector_store.save()
        print("Index saved successfully")
        
        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks)
        }
    
    def query(self, question: str, conversation_id: Optional[str] = None) -> QueryResponse:
        """Process a query"""
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Check cache
        cached_response = self.cache_service.get(question)
        if cached_response:
            cached_response['conversation_id'] = conversation_id
            return QueryResponse(**cached_response, cached=True)
        
        # Check if vector store has documents
        if self.vector_store.index.ntotal == 0:
            return QueryResponse(
                answer="⚠️ No documents have been indexed yet. Please upload and index documents first using the /index-documents endpoint.",
                sources=[],
                conversation_id=conversation_id,
                cached=False
            )
        
        print(f"\n{'='*60}")
        print(f"Processing query: {question}")
        print(f"Vector store contains: {self.vector_store.index.ntotal} vectors")
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(question)
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Search similar documents (increased k for better results)
        results = self.vector_store.similarity_search(
            query_embedding,
            k=settings.top_k_results
        )
        
        print(f"Found {len(results)} similar documents")
        
        # Debug: Print distances
        for i, (doc, distance) in enumerate(results):
            print(f"  Result {i+1}: distance={distance:.4f}, source={doc.metadata.get('source', 'Unknown')}")
        
        # If no results found
        if not results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the indexed documents to answer your question.",
                sources=[],
                conversation_id=conversation_id,
                cached=False
            )
        
        # Prepare context from top results
        context_parts = []
        for doc, distance in results:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"[Source: {source}{page_info}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        print(f"Context length: {len(context)} characters")
        
        # Get conversation history
        conversation_history = None
        if conversation_id and conversation_id in self.conversation_store:
            conversation_history = self.conversation_store[conversation_id]
            print(f"Using conversation history with {len(conversation_history)} messages")
        
        # Generate answer
        answer = self.llm_service.generate_answer(question, context, conversation_history)
        print(f"Generated answer length: {len(answer)} characters")
        
        # Prepare sources with more details
        sources = [
            SourceDocument(
                content=doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                page=doc.metadata.get("page")
            )
            for doc, distance in results
        ]
        
        print(f"Prepared {len(sources)} source documents")
        print(f"{'='*60}\n")
        
        # Create response
        response = QueryResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            cached=False
        )
        
        # Update conversation history
        if conversation_id:
            if conversation_id not in self.conversation_store:
                self.conversation_store[conversation_id] = []
            self.conversation_store[conversation_id].append({"role": "user", "content": question})
            self.conversation_store[conversation_id].append({"role": "assistant", "content": answer})
        
        # Cache response (without conversation_id to make cache reusable)
        cache_data = response.dict(exclude={"cached", "conversation_id"})
        self.cache_service.set(question, cache_data)
        
        return response