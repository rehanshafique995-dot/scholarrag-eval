from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/indexes")
INDEX_PATH = INDEX_DIR / "dense_index.faiss"
METADATA_PATH = INDEX_DIR / "dense_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    """Load all chunk JSON files."""
    all_chunks = []

    for chunk_file in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)

    return all_chunks


def build_dense_index() -> None:
    """Build FAISS dense vector index from all chunks."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    if not chunks:
        print("No chunks found in data/chunks/")
        return

    texts = [chunk["text"] for chunk in chunks]

    print(f"Loaded chunks: {len(texts)}")
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    METADATA_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Dense index created successfully.")
    print(f"Index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search top-k relevant chunks using dense retrieval."""
    model = SentenceTransformer(MODEL_NAME)

    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        results.append({
            "score": float(score),
            "paper_id": chunk.get("paper_id"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
        })

    return results


if __name__ == "__main__":
    build_dense_index()

    test_query = "What is retrieval augmented generation?"
    results = search(test_query, top_k=3)

    print("\nSearch results:")
    for result in results:
        print("=" * 80)
        print(f"Score: {result['score']}")
        print(f"Paper: {result['paper_id']} | Chunk: {result['chunk_id']}")
        print(result["text"][:700])