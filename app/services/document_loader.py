import os
from typing import List
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import get_settings

settings = get_settings()

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and extract text from PDF"""
        documents = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(file_path),
                            "page": page_num + 1
                        }
                    ))
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")
        return documents
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return [Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path)}
            )]
        except Exception as e:
            print(f"Error loading text file {file_path}: {e}")
            return []
    
    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """Load all documents from a directory"""
        all_documents = []
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if filename.endswith('.pdf'):
                all_documents.extend(self.load_pdf(file_path))
            elif filename.endswith('.txt'):
                all_documents.extend(self.load_text(file_path))
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)