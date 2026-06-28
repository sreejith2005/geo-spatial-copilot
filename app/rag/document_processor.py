import os
import re
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF Processor.
        
        Args:
            chunk_size: The target size of text chunks (in characters).
            chunk_overlap: The number of characters to overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain's text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdfs(self, directory_path: str) -> List[Document]:
        """
        Load all PDF files from the given directory.
        
        Args:
            directory_path: Path to the directory containing PDFs.
            
        Returns:
            A list of LangChain Document objects.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Directory not found: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        documents = []
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDFs found in directory: {directory_path}")
            return documents
            
        logger.info(f"Found {len(pdf_files)} PDFs in {directory_path}. Starting extraction...")
        
        for pdf_path in pdf_files:
            try:
                logger.info(f"Loading {pdf_path.name}")
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                # Ensure metadata contains relative paths or just filename to be safe
                for doc in docs:
                    doc.metadata["source_file"] = pdf_path.name
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {pdf_path.name}: {e}")
                
        logger.info(f"Loaded a total of {len(documents)} pages from PDFs.")
        return documents

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text extracted from PDF.
            
        Returns:
            Cleaned text.
        """
        if not text:
            return ""
        
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean and chunk the loaded documents.
        
        Args:
            documents: List of loaded Document objects.
            
        Returns:
            List of chunked Document objects.
        """
        logger.info("Cleaning document texts...")
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            if cleaned_content:  # Skip empty pages
                cleaned_docs.append(Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata
                ))
                
        logger.info(f"Splitting {len(cleaned_docs)} pages into chunks...")
        chunks = self.text_splitter.split_documents(cleaned_docs)
        
        # Add chunk index to metadata for tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            
        logger.info(f"Generated {len(chunks)} text chunks.")
        return chunks
