import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text", sort=True)
        text += f"\n\n--- PAGE {page_num} ---\n\n"
        text += page_text

    doc.close()
    return text


if __name__ == "__main__":
    pdf_path = "papers/pdfs/P001_rag_knowledge_intensive_nlp.pdf"
    output_path = "papers/parsed/P001_rag_knowledge_intensive_nlp.txt"

    text = extract_text_from_pdf(pdf_path)

    Path(output_path).write_text(text, encoding="utf-8")

    print("PDF parsed successfully.")
    print(f"Saved to: {output_path}")
    print(f"Characters extracted: {len(text)}")