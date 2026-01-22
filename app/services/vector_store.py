import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple
from langchain_core.documents import Document

class FAISSVectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Document] = []
        self.index_path = "storage/faiss_index"
        os.makedirs(self.index_path, exist_ok=True)
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and their embeddings to the index"""
        if len(documents) != len(embeddings):
            raise ValueError(f"Document count ({len(documents)}) doesn't match embedding count ({len(embeddings)})")
        
        self.documents.extend(documents)
        embeddings_float32 = embeddings.astype('float32')
        self.index.add(embeddings_float32)
        print(f"Added {len(documents)} documents. Total in index: {self.index.ntotal}")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            print("Warning: Vector store is empty!")
            return []
        
        # Ensure proper shape and type
        query_embedding = query_embedding.astype('float32')
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Limit k to available documents
        k = min(k, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k)
        
        print(f"Search results - distances: {distances[0]}, indices: {indices[0]}")
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))
            else:
                print(f"Warning: Invalid index {idx}, documents length: {len(self.documents)}")
        
        return results
    
    def save(self):
        """Save index and documents to disk"""
        try:
            faiss.write_index(self.index, f"{self.index_path}/faiss.index")
            with open(f"{self.index_path}/documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            print(f"Saved index with {self.index.ntotal} vectors and {len(self.documents)} documents")
        except Exception as e:
            print(f"Error saving index: {e}")
            raise
    
    def load(self):
        """Load index and documents from disk"""
        try:
            index_file = f"{self.index_path}/faiss.index"
            docs_file = f"{self.index_path}/documents.pkl"
            
            if not os.path.exists(index_file) or not os.path.exists(docs_file):
                print("No existing index found")
                return False
            
            self.index = faiss.read_index(index_file)
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            print(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False
    
    def clear(self):
        """Clear the index and documents"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        print("Index cleared")