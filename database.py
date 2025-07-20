import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.generativeai import GenerativeModel, configure
from config import GEMINI_API_KEY, CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME

# Configure Gemini API
configure(api_key=GEMINI_API_KEY)

# Custom Embedding Function for ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, task_type: str = "retrieval_document"):
        self.model_name = model_name
        self.task_type = task_type
        self.embedding_model = GenerativeModel(self.model_name)

    def __call__(self, input: Documents) -> Embeddings:
        # The Gemini API embed_content method expects a list of strings for content
        # For retrieval_document, you generally embed each document separately.
        embeddings_list = []
        for text_content in input:
            response = self.embedding_model.embed_content(
                model=self.model_name,
                content=text_content,
                task_type=self.task_type
            )
            embeddings_list.append(response.embedding)
        return embeddings_list

# Initialize ChromaDB Client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_chroma_collection(document_mode: bool = True):
    """
    Returns the ChromaDB collection, creating it if it doesn't exist.
    Uses a custom Gemini Embedding Function.
    """
    embedding_func = GeminiEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        task_type="retrieval_document" if document_mode else "retrieval_query"
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

# Global collection instance (can be initialized once)
chroma_collection = get_chroma_collection(document_mode=True)

def ingest_documents_to_chroma(documents: list[dict]):
    """Ingests documents into the ChromaDB collection."""
    ids = [doc["id"] for doc in documents]
    contents = [doc["content"] for doc in documents]
    metadatas = [doc.get("metadata", {}) for doc in documents]

    # Add documents to the collection
    # The embedding_function defined in the collection will be used automatically
    chroma_collection.add(
        documents=contents,
        metadatas=metadatas,
        ids=ids
    )
    return chroma_collection.count()

def query_chroma(query_text: str, n_results: int = 3) -> list[str]:
    """Queries ChromaDB for relevant documents."""
    # Ensure the embedding function is set for query mode if needed,
    # though ChromaDB handles this if you've set it up correctly with
    # separate functions for document/query. For simplicity, we assume
    # `get_chroma_collection` handles the `task_type` appropriately.

    # Generate query embedding
    query_embedding_func = GeminiEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        task_type="retrieval_query"
    )
    query_embedding = query_embedding_func([query_text])[0] # embedding_func expects a list

    results = chroma_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents']
    )
    return results['documents'][0] if results['documents'] else []