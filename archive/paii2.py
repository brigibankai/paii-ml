import argparse
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    """Abstract class to define the structure for embedding models."""
    @abstractmethod
    def embed_text(self, text):
        pass

class SentenceEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text):
        """Convert text into a vector embedding."""
        return self.model.encode([text])[0]


class VectorDatabase(ABC):
    """Abstract class for vector databases."""
    @abstractmethod
    def add_text(self, text, vector):
        pass

    @abstractmethod
    def search(self, query_vector, top_k=3):
        pass

class FaissDatabase(VectorDatabase):
    def __init__(self, dim=384, index_path="faiss_index.bin", text_data_path="text_data.txt"):
        self.dim = dim
        self.index = faiss.IndexFlatL2(self.dim)
        self.text_data = []
        self.index_path = index_path
        self.text_data_path = text_data_path
        self.load_data()

    def save_data(self):
        """Save FAISS index and text data to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.text_data_path, "w") as f:
            f.write("\n".join(self.text_data))
        print("✅ FAISS index and text data saved.")

    def load_data(self):
        """Load FAISS index and text data from disk if available."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print("✅ FAISS index loaded.")
        else:
            print("⚠️ No FAISS index found. Starting fresh.")

        if os.path.exists(self.text_data_path):
            with open(self.text_data_path, "r") as f:
                self.text_data = f.read().splitlines()
            print("✅ Text data loaded.")
        else:
            print("⚠️ No text data found. Starting fresh.")

    def add_text(self, text, vector):
        """Add text and vector to the FAISS index."""
        if text in self.text_data:
            print(f"⚠️ Duplicate entry detected: \"{text}\". Skipping...")
            return
        self.index.add(vector)
        self.text_data.append(text)
        self.save_data()
        print(f"✅ Added to database: {text}")

    def search(self, query_vector, top_k=3):
        """Search FAISS for the most similar results and return unique text matches."""
        distances, indices = self.index.search(query_vector, top_k)
        unique_results = {}
        min_dist, max_dist = min(distances[0]), max(distances[0]) if distances[0].size > 0 else (0, 1)

        # Normalize the scores
        if max_dist - min_dist < 1e-9:
            min_dist -= 1  # Shift min slightly so normalization works

        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.text_data):
                text_entry = self.text_data[i]
                normalized_score = 1 - ((dist - min_dist) / (max_dist - min_dist + 1e-9))  # Avoid division by zero
                normalized_score = max(0.01, normalized_score)  # Ensure no score is exactly 0.0000
                if text_entry not in unique_results:
                    unique_results[text_entry] = normalized_score

        sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
        final_texts, final_scores = zip(*sorted_results) if sorted_results else ([], [])
        return final_texts, final_scores


class PdfProcessor:
    def __init__(self, faiss_db: FaissDatabase):
        self.faiss_db = faiss_db

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()

    def chunk_text(self, text, max_chunk_size=500):
        """Splits text into paragraph-based chunks."""
        paragraphs = text.split("\n")  # Split on newlines
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue  # Skip empty lines

            # Check if adding this paragraph exceeds chunk size
            if current_length + len(para) > max_chunk_size:
                chunks.append(" ".join(current_chunk))  # Store completed chunk
                current_chunk = [para]  # Start new chunk
                current_length = len(para)
            else:
                current_chunk.append(para)
                current_length += len(para)

        if current_chunk:
            chunks.append(" ".join(current_chunk))  # Store last chunk
        return chunks

    def process_pdf(self, pdf_path):
        """Extract and add PDF text chunks to FAISS."""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        for chunk in chunks:
            self.faiss_db.add_text(chunk)


class PAIISystem:
    def __init__(self, embedding_model: EmbeddingModel, vector_db: VectorDatabase):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def embed_text(self, text):
        """Convert text into a vector embedding using the embedding model."""
        return self.embedding_model.embed_text(text)

    def add_to_db(self, text):
        """Convert text to vector and add it to the vector database."""
        vector = np.array([self.embed_text(text)], dtype="float32")
        self.vector_db.add_text(text, vector)

    def search_db(self, query, top_k=3):
        """Search the vector database and return similar text."""
        query_vector = np.array([self.embed_text(query)], dtype="float32").reshape(1, -1)
        return self.vector_db.search(query_vector, top_k)

    def process_input(self):
        """Handles CLI input."""
        parser = argparse.ArgumentParser(description="PAII Systems CLI")
        parser.add_argument("--query", "-q", type=str, help="Enter query")
        parser.add_argument("--add", "-a", type=str, help="Add text to FAISS")
        parser.add_argument("--pdf", "-p", type=str, help="Add PDF text to FAISS")
        return parser.parse_args()


if __name__ == "__main__":
    embedding_model = SentenceEmbeddingModel()
    vector_db = FaissDatabase()
    system = PAIISystem(embedding_model, vector_db)

    args = system.process_input()

    if args.add:
        system.add_to_db(args.add)
    elif args.pdf:
        pdf_processor = PdfProcessor(vector_db)
        pdf_processor.process_pdf(args.pdf)
    elif args.query:
        results, distances = system.search_db(args.query)
        for i, (result, dist) in enumerate(zip(results, distances)):
            print(f"{i+1}. {result} (Score: {dist:.4f})")
    else:
        print("Please provide either --add, --query, or --pdf.")
