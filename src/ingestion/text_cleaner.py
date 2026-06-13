from pathlib import Path
import re


PARSED_DIR = Path("papers/parsed")
CLEANED_DIR = Path("data/cleaned")


def clean_text(text: str) -> str:
    """Clean parsed PDF text."""
    # Remove page markers like --- PAGE 1 ---
    text = re.sub(r"\n\n--- PAGE \d+ ---\n\n", "\n", text)

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace 3+ newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before newlines
    text = re.sub(r" +\n", "\n", text)

    return text.strip()


def clean_all_parsed_files() -> None:
    """Clean all parsed .txt files and save cleaned versions."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(PARSED_DIR.glob("*.txt"))

    if not txt_files:
        print("No parsed text files found in papers/parsed/")
        return

    for txt_file in txt_files:
        raw_text = txt_file.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = clean_text(raw_text)

        output_file = CLEANED_DIR / txt_file.name
        output_file.write_text(cleaned_text, encoding="utf-8")

        print(f"Cleaned: {txt_file.name}")
        print(f"Raw chars: {len(raw_text)} | Cleaned chars: {len(cleaned_text)}")
        print("-" * 50)


if __name__ == "__main__":
    clean_all_parsed_files()