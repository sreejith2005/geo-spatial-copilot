import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to sys.path so we can import 'app'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.rag.document_processor import PDFProcessor
from app.rag.indexer import QdrantIndexer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Setup paths relative to the project root
    documents_dir = project_root / "data" / "documents"
    interim_dir = project_root / "data" / "interim" / "chunks"
    
    # Ensure intermediate directory exists
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting ingestion pipeline...")
    
    # 1. Initialize Processor
    processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    
    # 2. Load PDFs
    docs = processor.load_pdfs(str(documents_dir))
    if not docs:
        logger.warning("No documents loaded. Exiting.")
        return
        
    # 3. Clean and Chunk
    chunks = processor.process_documents(docs)
    
    # 4. Save intermediate chunks to JSONL
    chunks_file = interim_dir / "chunks.jsonl"
    logger.info(f"Saving intermediate chunks to {chunks_file}")
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    # 5. Initialize Indexer and Index in Qdrant
    logger.info("Initializing Qdrant Indexer...")
    try:
        indexer = QdrantIndexer(collection_name="documents")
        indexer.index_chunks(chunks)
    except Exception as e:
        logger.error(f"Failed to index chunks: {e}")
        return
        
    # 6. Test Retrieval
    test_query = "What are the major flood risks and resilience strategies in India?"
    logger.info("\n=== Running Test Retrieval ===")
    results = indexer.search(query=test_query, limit=3)
    
    print(f"\nQuery: {test_query}\n")
    for i, res in enumerate(results):
        print(f"Result {i+1} (Score: {res['score']:.4f})")
        print(f"Source: {res['metadata'].get('source_file', 'Unknown')} - Page: {res['metadata'].get('page', 'Unknown')}")
        print(f"Excerpt: {res['text'][:200]}...\n")
        
    logger.info("Ingestion pipeline completed successfully.")

if __name__ == "__main__":
    main()
