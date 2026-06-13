from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi


CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/indexes")
BM25_METADATA_PATH = INDEX_DIR / "bm25_metadata.json"


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def load_chunks() -> list[dict]:
    """Load all chunk JSON files."""
    all_chunks = []

    for chunk_file in sorted(CHUNKS_DIR.glob("*_chunks.json")):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)

    return all_chunks


def build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Build BM25 index from chunks."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    if not chunks:
        raise FileNotFoundError("No chunks found in data/chunks/. Run chunker.py first.")

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    BM25_METADATA_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("BM25 index built successfully.")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Metadata saved to: {BM25_METADATA_PATH}")

    return bm25, chunks


def search(query: str, bm25: BM25Okapi, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Search top-k relevant chunks using BM25."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "paper_id": chunk.get("paper_id"),
            "chunk_id": chunk.get("chunk_id"),
            "text": chunk.get("text"),
        })

    return results


if __name__ == "__main__":
    bm25, chunks = build_bm25_index()

    test_query = "What metrics are used to evaluate RAG systems?"
    results = search(test_query, bm25, chunks, top_k=5)

    print("\nBM25 Search Results:")
    for result in results:
        print("=" * 80)
        print(f"Score: {result['score']}")
        print(f"Paper: {result['paper_id']} | Chunk: {result['chunk_id']}")
        print(result["text"][:700])