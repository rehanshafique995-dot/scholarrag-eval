from pathlib import Path
import json


CLEANED_DIR = Path("data/cleaned")
CHUNKS_DIR = Path("data/chunks")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "start_char": start,
                "end_char": min(end, len(text)),
            })
            chunk_id += 1

        start += chunk_size - overlap

    return chunks


def chunk_all_cleaned_files() -> None:
    """Create chunks for all cleaned paper text files."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(CLEANED_DIR.glob("*.txt"))

    if not txt_files:
        print("No cleaned text files found in data/cleaned/")
        return

    for txt_file in txt_files:
        paper_id = txt_file.stem.split("_")[0]
        text = txt_file.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text(text)

        for chunk in chunks:
            chunk["paper_id"] = paper_id
            chunk["source_file"] = txt_file.name

        output_file = CHUNKS_DIR / f"{txt_file.stem}_chunks.json"

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Chunked: {txt_file.name}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Saved: {output_file}")
        print("-" * 50)


if __name__ == "__main__":
    chunk_all_cleaned_files()