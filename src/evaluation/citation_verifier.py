from difflib import SequenceMatcher
from src.retrieval.section_hybrid_retriever import section_hybrid_search
from src.retrieval.section_dense_retriever import build_section_dense_index
from src.retrieval.section_bm25_retriever import build_section_bm25_index


def similarity_score(text_a: str, text_b: str) -> float:
    """Calculate similarity between two text strings."""
    return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


def verify_claim_against_evidence(claim: str, evidence_chunks: list[dict]) -> dict:
    """Verify one claim against retrieved evidence chunks."""
    best_match = None
    best_score = 0.0

    for chunk in evidence_chunks:
        score = similarity_score(claim, chunk["text"])

        if score > best_score:
            best_score = score
            best_match = chunk

    if best_score >= 0.35:
        status = "supported"
    elif best_score >= 0.20:
        status = "weakly_supported"
    else:
        status = "unsupported"

    return {
        "claim": claim,
        "support_status": status,
        "best_score": round(best_score, 4),
        "best_paper_id": best_match.get("paper_id") if best_match else None,
        "best_section": best_match.get("section") if best_match else None,
        "best_chunk_id": best_match.get("chunk_id") if best_match else None,
    }


def verify_answer_claims(claims: list[str], evidence_chunks: list[dict]) -> list[dict]:
    """Verify multiple claims against retrieved evidence."""
    return [
        verify_claim_against_evidence(claim, evidence_chunks)
        for claim in claims
    ]


if __name__ == "__main__":
    query = "Which metrics are used to evaluate RAG systems?"

    print("Building section-aware dense index...")
    build_section_dense_index()

    print("Building section-aware BM25 index...")
    bm25, bm25_chunks = build_section_bm25_index()

    evidence = section_hybrid_search(
        query=query,
        bm25=bm25,
        bm25_chunks=bm25_chunks,
        top_k=5,
    )

    test_claims = [
        "RAG systems can be evaluated using faithfulness and answer relevance.",
        "Citation quality is important for evaluating generated answers.",
        "RAG systems always use supervised fine-tuning as the main retrieval method.",
    ]

    verification_results = verify_answer_claims(test_claims, evidence)

    print("\nCitation Verification Results")
    print("=" * 80)

    for result in verification_results:
        print(f"\nClaim: {result['claim']}")
        print(f"Status: {result['support_status']}")
        print(f"Best score: {result['best_score']}")
        print(
            f"Best evidence: {result['best_paper_id']} | "
            f"{result['best_section']} | "
            f"Chunk {result['best_chunk_id']}"
        )