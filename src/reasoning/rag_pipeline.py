from pathlib import Path
import re

from src.retrieval.section_dense_retriever import build_section_dense_index
from src.retrieval.section_bm25_retriever import build_section_bm25_index
from src.retrieval.section_hybrid_retriever import section_hybrid_search

from src.generation.answer_generator import generate_extractive_answer
from src.evaluation.citation_verifier import verify_answer_claims
from src.evaluation.unsupported_claim_detector import detect_unsupported_claims


OUTPUT_DIR = Path("experiments/generation_results")
OUTPUT_PATH = OUTPUT_DIR / "rag_pipeline_result.txt"


def extract_claims_from_answer(answer: str) -> list[str]:
    """Extract simple claims from generated answer."""
    claims = []

    for line in answer.splitlines():
        line = line.strip()

        if re.match(r"^\d+\.", line):
            clean_line = re.sub(r"^\d+\.\s*", "", line)
            claims.append(clean_line)

    return claims


def run_rag_pipeline(query: str, top_k: int = 5) -> dict:
    """Run retrieval, answer generation, citation verification, and unsupported-claim detection."""
    print("Building section-aware dense index...")
    build_section_dense_index()

    print("Building section-aware BM25 index...")
    bm25, bm25_chunks = build_section_bm25_index()

    print("Retrieving evidence...")
    evidence = section_hybrid_search(
        query=query,
        bm25=bm25,
        bm25_chunks=bm25_chunks,
        top_k=top_k,
    )

    print("Generating answer...")
    answer = generate_extractive_answer(query, evidence)

    print("Extracting claims...")
    claims = extract_claims_from_answer(answer)

    print("Verifying claims...")
    verification_results = verify_answer_claims(claims, evidence)

    print("Detecting unsupported claims...")
    support_summary = detect_unsupported_claims(verification_results)

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence,
        "claims": claims,
        "verification_results": verification_results,
        "support_summary": support_summary,
    }


def save_pipeline_result(result: dict) -> None:
    """Save complete pipeline result locally."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write("ScholarRAG-Eval Pipeline Result\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Query:\n{result['query']}\n\n")

        f.write("Generated Answer:\n")
        f.write("-" * 80 + "\n")
        f.write(result["answer"])
        f.write("\n\n")

        f.write("Extracted Claims:\n")
        f.write("-" * 80 + "\n")
        for claim in result["claims"]:
            f.write(f"- {claim}\n")

        f.write("\nVerification Results:\n")
        f.write("-" * 80 + "\n")
        for item in result["verification_results"]:
            f.write(
                f"Claim: {item['claim']}\n"
                f"Status: {item['support_status']}\n"
                f"Best score: {item['best_score']}\n"
                f"Evidence: {item['best_paper_id']} | "
                f"{item['best_section']} | Chunk {item['best_chunk_id']}\n\n"
            )

        f.write("Support Summary:\n")
        f.write("-" * 80 + "\n")
        summary = result["support_summary"]
        f.write(f"Overall status: {summary['overall_status']}\n")
        f.write(f"Total claims: {summary['total_claims']}\n")
        f.write(f"Supported: {summary['supported_count']}\n")
        f.write(f"Weakly supported: {summary['weakly_supported_count']}\n")
        f.write(f"Unsupported: {summary['unsupported_count']}\n")

    print(f"Pipeline result saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    query = "Which metrics are used to evaluate RAG systems?"

    result = run_rag_pipeline(query=query, top_k=5)

    print("\nFinal Answer")
    print("=" * 80)
    print(result["answer"])

    print("\nSupport Summary")
    print("=" * 80)
    print(result["support_summary"])

    save_pipeline_result(result)