import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import os

class QdrantIndexer:
    def __init__(self, 
                 collection_name: str = "documents", 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 qdrant_url: str = None):
        """
        Initialize the Qdrant Indexer.
        
        Args:
            collection_name: Name of the Qdrant collection to create/use.
            embedding_model_name: HuggingFace model name for sentence-transformers.
            qdrant_url: URL to the Qdrant instance.
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        
        logger.info(f"Loading embedding model: {embedding_model_name}")
        # Add HF token if present to avoid warnings/rate limits
        hf_token = os.environ.get("HF_TOKEN")
        self.embedding_model = SentenceTransformer(embedding_model_name, token=hf_token)
        
        # Avoid deprecated get_sentence_embedding_dimension()
        test_embedding = self.embedding_model.encode("test")
        self.vector_size = len(test_embedding)
        logger.info(f"Model loaded. Vector dimension is {self.vector_size}")
        
        logger.info(f"Connecting to Qdrant at {self.qdrant_url}")
        self.client = QdrantClient(url=self.qdrant_url)
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the target collection exists in Qdrant, create if not."""
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        
        if not exists:
            logger.info(f"Collection '{self.collection_name}' not found. Creating new collection...")
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
                logger.info(f"Collection '{self.collection_name}' created successfully.")
            except Exception as e:
                logger.warning(f"Could not create collection (might have been created by another worker): {e}")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def index_chunks(self, chunks: List[Document]):
        """
        Create embeddings for the chunks and upload them to Qdrant.
        
        Args:
            chunks: List of chunked Document objects.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk.page_content for chunk in chunks]
        
        # We can encode in batches, sentence-transformers handles batching internally by default
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        logger.info("Uploading vectors to Qdrant...")
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Ensure metadata exists
            payload = chunk.metadata.copy()
            payload["text"] = chunk.page_content
            
            # Using chunk_id from metadata or fallback to index if missing
            point_id = payload.get("chunk_id", i)
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )
            
        # Upload in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Successfully indexed {len(points)} chunks into Qdrant.")

    def search(self, query: str, limit: int = 5) -> List[dict]:
        """
        Perform a similarity search on the indexed chunks.
        
        Args:
            query: The search query string.
            limit: Number of results to return.
            
        Returns:
            List of dictionaries containing matching chunks and their scores.
        """
        logger.info(f"Searching for: '{query}'")
        query_vector = self.embedding_model.encode(query).tolist()
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )
        
        results = []
        for scored_point in search_result.points:
            payload = scored_point.payload or {}
            results.append({
                "score": scored_point.score,
                "text": payload.get("text", ""),
                "metadata": {k: v for k, v in payload.items() if k != "text"}
            })
            
        return results
