from pathlib import Path
import re


CLEANED_DIR = Path("data/cleaned")
SECTION_DIR = Path("data/sections")


SECTION_PATTERNS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "limitations",
    "conclusion",
    "references",
]


def detect_sections(text: str) -> dict:
    """Detect major paper sections using heading patterns."""
    matches = []

    for pattern in SECTION_PATTERNS:
        regex = rf"(?im)^\s*(\d+\.?\s*)?{re.escape(pattern)}\s*$"
        for match in re.finditer(regex, text):
            matches.append((match.start(), pattern.title()))

    matches = sorted(matches, key=lambda x: x[0])

    sections = {}

    for i, (start_pos, section_name) in enumerate(matches):
        end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        sections[section_name] = text[start_pos:end_pos].strip()

    return sections


def save_sections_for_all_files() -> None:
    """Detect and save sections for all cleaned papers."""
    SECTION_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(CLEANED_DIR.glob("*.txt"))

    if not txt_files:
        print("No cleaned text files found in data/cleaned/")
        return

    for txt_file in txt_files:
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        sections = detect_sections(text)

        output_file = SECTION_DIR / f"{txt_file.stem}_sections.txt"

        with output_file.open("w", encoding="utf-8") as f:
            for section_name, section_text in sections.items():
                f.write(f"\n\n===== {section_name} =====\n\n")
                f.write(section_text)

        print(f"Processed: {txt_file.name}")
        print(f"Sections found: {list(sections.keys())}")
        print(f"Saved: {output_file}")
        print("-" * 50)


if __name__ == "__main__":
    save_sections_for_all_files()