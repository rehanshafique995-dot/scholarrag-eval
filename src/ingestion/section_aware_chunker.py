from pathlib import Path
import json
import re


SECTIONS_DIR = Path("data/sections")
OUTPUT_DIR = Path("data/section_chunks")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def split_sections(section_text: str) -> list[tuple[str, str]]:
    """Split section detector output into section name + section body."""
    pattern = r"===== (.*?) ====="
    parts = re.split(pattern, section_text)

    sections = []

    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        if body:
            sections.append((section_name, body))

    return sections


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Create overlapping character chunks."""
    chunks = []
    start = 0
    chunk_id = 1

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append({
                "local_chunk_id": chunk_id,
                "text": chunk,
                "start_char": start,
                "end_char": min(end, len(text)),
            })
            chunk_id += 1

        start += chunk_size - overlap

    return chunks


def create_section_aware_chunks() -> None:
    """Create section-aware chunks for all section files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    section_files = sorted(SECTIONS_DIR.glob("*_sections.txt"))

    if not section_files:
        print("No section files found in data/sections/. Run section_detector.py first.")
        return

    for section_file in section_files:
        paper_id = section_file.stem.split("_")[0]
        raw_text = section_file.read_text(encoding="utf-8", errors="ignore")

        sections = split_sections(raw_text)

        all_chunks = []
        global_chunk_id = 1

        for section_name, section_body in sections:
            section_chunks = chunk_text(section_body)

            for chunk in section_chunks:
                chunk["global_chunk_id"] = global_chunk_id
                chunk["paper_id"] = paper_id
                chunk["section"] = section_name
                chunk["source_file"] = section_file.name

                all_chunks.append(chunk)
                global_chunk_id += 1

        output_file = OUTPUT_DIR / f"{paper_id}_section_chunks.json"

        output_file.write_text(
            json.dumps(all_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(f"Processed: {section_file.name}")
        print(f"Sections found: {len(sections)}")
        print(f"Section-aware chunks created: {len(all_chunks)}")
        print(f"Saved: {output_file}")
        print("-" * 60)


if __name__ == "__main__":
    create_section_aware_chunks()