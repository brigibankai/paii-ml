"""
PDF extraction and chunking utilities.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PdfProcessor:
    """Extract and chunk text from PDF files."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize PDF processor.
        
        Parameters
        ----------
        chunk_size : int
            Target chunk size in characters.
        chunk_overlap : int
            Overlap between consecutive chunks in characters.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Parameters
        ----------
        pdf_path : str
            Path to PDF file.
        
        Returns
        -------
        str
            Extracted text from all pages.
        
        Raises
        ------
        FileNotFoundError
            If PDF does not exist.
        ImportError
            If PyMuPDF is not installed.
        """
        try:
            import fitz
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Extracting text from {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            text_pages = []
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                text_pages.append(text)
            
            text = "\n".join(text_pages)
            doc.close()
            
            logger.info(f"Extracted {len(text)} characters from {len(text_pages)} pages")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Failed to extract PDF: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata.
        
        Parameters
        ----------
        text : str
            Text to chunk.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of chunks with structure: {
                "text": str,
                "chunk_id": int,
                "start_char": int,
                "end_char": int
            }
        """
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end
                })
                chunk_id += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        logger.info(f"Chunked {len(text)} characters into {len(chunks)} chunks")
        return chunks
    
    def process(self, pdf_path: str, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract text from PDF and return chunks with metadata.
        
        Parameters
        ----------
        pdf_path : str
            Path to PDF file.
        source_name : Optional[str]
            Human-readable name for the source (default: filename).
        
        Returns
        -------
        List[Dict[str, Any]]
            List of chunks ready for embedding, each with:
            {
                "text": str,
                "source": str,
                "chunk_id": int,
                "start_char": int,
                "end_char": int
            }
        """
        text = self.extract_text(pdf_path)
        chunks = self.chunk_text(text)
        
        if source_name is None:
            source_name = Path(pdf_path).name
        
        # Add source metadata
        for chunk in chunks:
            chunk["source"] = source_name
        
        return chunks


if __name__ == "__main__":
    print("PDF processor module")

