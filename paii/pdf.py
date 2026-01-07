import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from paii.utils import chunk_by_paragraphs, clean_text

logger = logging.getLogger(__name__)


class PdfProcessor:
    """Extract and chunk text from PDF files with paragraph-aware chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_pages(self, pdf_path: str) -> List[str]:
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
            pages = [page.get_text("text") for page in doc]
            doc.close()
            logger.info(f"Extracted text from {len(pages)} pages")
            return pages
        except Exception as e:
            logger.error(f"Failed to extract PDF: {e}")
            raise

    def chunk_page(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        # Preserve paragraph separators for paragraph-aware chunking.
        if not text or not text.strip():
            return []

        # Normalize line endings but keep paragraph breaks (empty line)
        import re
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[Dict[str, Any]] = []
        char_cursor = 0
        for cid, p in enumerate(paras):
            start = text.find(p, char_cursor)
            if start == -1:
                start = char_cursor
            end = start + len(p)
            char_cursor = end

            chunks.append({
                "text": p,
                "chunk_id": cid,
                "start_char": start,
                "end_char": end,
                "page": page_num,
            })

        logger.debug(f"Chunked page {page_num} into {len(chunks)} chunks")
        return chunks

    def process(self, pdf_path: str, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
        pages = self.extract_text_pages(pdf_path)

        if source_name is None:
            source_name = Path(pdf_path).name

        all_chunks: List[Dict[str, Any]] = []
        for page_idx, page_text in enumerate(pages, start=1):
            page_chunks = self.chunk_page(page_text, page_idx)
            for c in page_chunks:
                # Add provenance and text hash
                c["source"] = source_name
                c["text_hash"] = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
                all_chunks.append(c)

        logger.info(f"Processed {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
        for chunk in chunks:
            chunk["source"] = source_name
        
        return chunks


if __name__ == "__main__":
    print("PDF processor module")

