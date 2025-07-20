from pydantic import BaseModel
from typing import List, Optional

class Document(BaseModel):
    id: str
    content: str
    metadata: Optional[dict] = None

class IngestRequest(BaseModel):
    documents: List[Document]

class QueryRequest(BaseModel):
    query: str
    n_results: int = 3 # Number of top relevant documents to retrieve

class RagResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[str]