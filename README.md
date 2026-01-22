# Medical Policy RAG Chatbot

A powerful Retrieval-Augmented Generation (RAG) chatbot designed to parse, index, and query medical policy documents. Built with **FastAPI**, **LangChain**, **Groq (Llama 3)**, and **ChromaDB**.

## 🚀 Features

*   **📄 Document Ingestion**: Automatically processes and indexes PDF medical policy documents.
*   **🔍 Semantic Search**: Uses high-performance embeddings (`all-MiniLM-L6-v2`) and ChromaDB for accurate context retrieval.
*   **🤖 Advanced LLM Integration**: Leverages Groq's super-fast API with Llama 3 for intelligent, context-aware responses.
*   **⚡ High Performance Caching**: Redis integration for caching diverse query results to reduce latency and API costs.
*   **🛠️ Modern API**: Built on FastAPI with automatic Swagger/OpenAPI documentation.

## 🛠️ Tech Stack

*   **Backend**: Python, FastAPI
*   **LLM**: Llama 3 (via Groq)
*   **Vector Query**: ChromaDB
*   **Embeddings**: SentenceTransformers (HuggingFace)
*   **Orchestration**: LangChain
*   **Caching**: Redis
*   **Containerization**: Docker (for Redis)

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
*   [Python 3.9+](https://www.python.org/downloads/)
*   [Docker Desktop](https://www.docker.com/products/docker-desktop/)
*   [Groq API Key](https://console.groq.com/keys)

## 🏗️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd RAG_Chatbot_Medical_Policy_
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration**
    Create a `.env` file in the root directory and add your credentials:
    ```env
    # Groq API
    GROQ_API_KEY="your_groq_api_key_here"

    # Redis
    REDIS_HOST=localhost
    REDIS_PORT=6379
    REDIS_DB=0
    
    # Application Config
    CHUNK_SIZE=500
    CHUNK_OVERLAP=50
    TOP_K_RESULTS=3
    CACHE_EXPIRY=3600
    ```

5.  **Start Redis**
    Use Docker to spin up a Redis instance:
    ```bash
    docker-compose up -d
    ```

6.  **Run the Application**
    ```bash
    uvicorn app.main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

## 📖 Usage

### API Documentation
Once the server is running, navigate to `http://localhost:8000/docs` to access the interactive Swagger UI.

### Ingest Documents
Place your medical policy PDFs in the `Data/` directory (or use an upload endpoint if configured) to perform initial indexing.

### Chat with the Bot
Send a POST request to the chat endpoint to query your documents.

## 📸 API Demonstration

### 1. API Query
Here is an example of sending a query to the chatbot via the Swagger UI.

![API Query Screenshot](assets/api_query.png)
*(Place your screenshot here: Create an `assets` folder and save the image as `api_query.png`)*

### 2. API Response
The chatbot retrieves relevant context and answers based on the medical policy.

![API Response Screenshot](assets/api_response.png)
*(Place your screenshot here: Save the image as `api_response.png`)*

## 🧠 Approach & Architecture

1.  **Ingestion**: 
    -   Medical PDFs are loaded using `PyPDFLoader`.
    -   Text is split into chunks (Size: 500, Overlap: 50) to maintain context.
2.  **Embedding**: 
    -   Chunks are converted into vector embeddings using `SentenceTransformers`.
3.  **Storage**: 
    -   Vectors are stored in **ChromaDB** for efficient similarity search.
4.  **Retrieval (RAG)**: 
    -   User queries are embedded and compared against stored vectors.
    -   Top `k` most similar chunks are retrieved.
5.  **Generation**: 
    -   Retrieved context + User Query are sent to **Llama 3 (Groq)**.
    -   The LLM generates a precise answer based *only* on the provided context.
6.  **Caching**: 
    -   Responses are cached in **Redis** with a TTL to speed up repeated queries.

## 📂 Project Structure

```
RAG_Chatbot_Medical_Policy_/
├── app/
│   ├── api/            # Route definitions
│   ├── services/       # Business logic (LLM, RAG, Document Loader, Cache service, Vectore store)
│   ├── config.py       # Configuration management
│   ├── models.py       # Pydantic data models
│   └── main.py         # App entry point
├── Data/documents      # Directory for input PDF documents
├── storage/            # FAISS persistence directory
├── docker-compose.yml  # Redis container config
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```
