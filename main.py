from fastapi import FastAPI, HTTPException, status
from google.generativeai import GenerativeModel, configure
from pydantic import BaseModel
from typing import List, Optional
import os

from config import GEMINI_API_KEY, GENERATION_MODEL_NAME
from database import ingest_documents_to_chroma, query_chroma, chroma_collection
from models import Document, IngestRequest, QueryRequest, RagResponse

# Configure Gemini API
configure(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="RAG API with Gemini, ChromaDB, FastAPI",
    description="A simple Retrieval Augmented Generation (RAG) API."
)

# Initialize Gemini LLM for generation
try:
    gemini_llm = GenerativeModel(GENERATION_MODEL_NAME)
except Exception as e:
    print(f"Error initializing Gemini LLM: {e}")
    gemini_llm = None # Handle this gracefully if API key is invalid or model unavailable


@app.on_event("startup")
async def startup_event():
    # Optional: Load initial data from a file on startup
    # For production, you'd have a separate ingestion process or endpoint.
    if not chroma_collection.count():
        print("ChromaDB collection is empty. Attempting to ingest sample data...")
        try:
            with open("data/documents.txt", "r", encoding="utf-8") as f:
                sample_text = f.read()
            
            # Simple chunking for demonstration (you'd use a more robust chunker)
            chunks = [
                {"id": f"doc_{i}", "content": chunk.strip()}
                for i, chunk in enumerate(sample_text.split("\n\n")) if chunk.strip()
            ]
            
            if chunks:
                count = ingest_documents_to_chroma(chunks)
                print(f"Successfully ingested {count} sample documents.")
            else:
                print("No content found in data/documents.txt to ingest.")

        except FileNotFoundError:
            print("Warning: data/documents.txt not found. Please create it for sample data.")
        except Exception as e:
            print(f"Error during initial data ingestion: {e}")

@app.post("/ingest", response_model=dict, status_code=status.HTTP_201_CREATED)
async def ingest_documents(request: IngestRequest):
    """
    Ingest new documents into the RAG system.
    """
    try:
        documents_to_add = []
        for doc in request.documents:
            documents_to_add.append(doc.model_dump()) # Pydantic v2 .model_dump()
        
        count = ingest_documents_to_chroma(documents_to_add)
        return {"message": f"Successfully ingested {count} documents."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest documents: {str(e)}"
        )

@app.post("/query", response_model=RagResponse)
async def query_rag(request: QueryRequest):
    """
    Perform a RAG query: retrieve relevant documents and generate an answer using Gemini.
    """
    if gemini_llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini LLM not initialized. Check API key and configuration."
        )

    try:
        # 1. Retrieve relevant documents
        retrieved_docs = query_chroma(request.query, request.n_results)

        if not retrieved_docs:
            return RagResponse(
                query=request.query,
                answer="I could not find any relevant information in my knowledge base.",
                retrieved_documents=[]
            )

        # 2. Augment prompt with retrieved context
        context = "\n\n".join(retrieved_docs)
        prompt = f"""
        You are a helpful assistant. Answer the following question concisely and directly,
        based *only* on the provided context. If the answer cannot be found in the context,
        state that you don't know or that the information is not available.

        Context:
        {context}

        Question: {request.query}

        Answer:
        """

        # 3. Generate answer using Gemini
        response = gemini_llm.generate_content(prompt)
        
        # Access the text from the response object, handling potential errors
        answer_text = response.text
        if not answer_text:
            answer_text = "I couldn't generate an answer based on the provided context."

        return RagResponse(
            query=request.query,
            answer=answer_text,
            retrieved_documents=retrieved_docs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process RAG query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Try to count documents in ChromaDB to verify connection
        _ = chroma_collection.count()
        # Try to make a dummy call to Gemini to verify connectivity
        if gemini_llm:
            _ = gemini_llm.generate_content("hello", safety_settings={'HARASSMENT':'BLOCK_NONE'})
        return {"status": "ok", "message": "API and dependencies are healthy."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

# Optional: Endpoint to get current document count
@app.get("/document_count", response_model=dict)
async def get_document_count():
    """Returns the number of documents in the ChromaDB collection."""
    try:
        count = chroma_collection.count()
        return {"document_count": count}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document count: {str(e)}"
        )