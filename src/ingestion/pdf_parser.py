from pathlib import Path
import fitz  # PyMuPDF


PDF_DIR = Path("papers/pdfs")
OUTPUT_DIR = Path("papers/parsed")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a single PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text("text", sort=True)
        full_text += f"\n\n--- PAGE {page_number} ---\n\n"
        full_text += page_text

    doc.close()
    return full_text


def parse_all_pdfs() -> None:
    """Parse all PDF files from papers/pdfs and save text files in papers/parsed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in papers/pdfs/")
        return

    for pdf_file in pdf_files:
        output_file = OUTPUT_DIR / f"{pdf_file.stem}.txt"

        print(f"Parsing: {pdf_file.name}")
        text = extract_text_from_pdf(pdf_file)

        output_file.write_text(text, encoding="utf-8")

        print(f"Saved: {output_file}")
        print(f"Characters extracted: {len(text)}")
        print("-" * 50)


if __name__ == "__main__":
    parse_all_pdfs()