from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi


SECTION_CHUNKS_DIR = Path("data/section_chunks")
INDEX_DIR = Path("data/indexes")
METADATA_PATH = INDEX_DIR / "section_bm25_metadata.json"


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def load_section_chunks() -> list[dict]:
    """Load all section-aware chunk JSON files."""
    all_chunks = []

    for chunk_file in sorted(SECTION_CHUNKS_DIR.glob("*_section_chunks.json")):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        all_chunks.extend(chunks)

    return all_chunks


def get_chunk_id(chunk: dict) -> int:
    """Return a consistent chunk id."""
    return chunk.get("global_chunk_id") or chunk.get("chunk_id") or chunk.get("local_chunk_id")


def build_section_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """Build BM25 index from section-aware chunks."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_section_chunks()

    if not chunks:
        raise FileNotFoundError(
            "No section-aware chunks found in data/section_chunks/. "
            "Run section_aware_chunker.py first."
        )

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    METADATA_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Section-aware BM25 index built successfully.")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Metadata saved to: {METADATA_PATH}")

    return bm25, chunks


def search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Search section-aware chunks using BM25."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for idx in top_indices:
        chunk = chunks[idx]

        results.append({
            "score": float(scores[idx]),
            "paper_id": chunk.get("paper_id"),
            "section": chunk.get("section"),
            "chunk_id": get_chunk_id(chunk),
            "text": chunk.get("text"),
        })

    return results


if __name__ == "__main__":
    bm25, chunks = build_section_bm25_index()

    query = "Which metrics are used to evaluate RAG systems?"
    results = search(query, bm25, chunks, top_k=5)

    print(f"\nQUERY: {query}")
    print("=" * 100)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(
            f"Paper: {result['paper_id']} | "
            f"Section: {result['section']} | "
            f"Chunk: {result['chunk_id']}"
        )
        print(result["text"][:700].replace("\n", " "))
        print("-" * 100)