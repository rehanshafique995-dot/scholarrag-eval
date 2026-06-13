from src.evaluation.citation_verifier import verify_answer_claims
from src.retrieval.section_dense_retriever import build_section_dense_index
from src.retrieval.section_bm25_retriever import build_section_bm25_index
from src.retrieval.section_hybrid_retriever import section_hybrid_search


def detect_unsupported_claims(verification_results: list[dict]) -> dict:
    """Summarize unsupported and weakly supported claims."""
    supported = []
    weakly_supported = []
    unsupported = []

    for result in verification_results:
        status = result["support_status"]

        if status == "supported":
            supported.append(result)
        elif status == "weakly_supported":
            weakly_supported.append(result)
        else:
            unsupported.append(result)

    total_claims = len(verification_results)

    if total_claims == 0:
        overall_status = "no_claims_found"
    elif len(unsupported) > 0:
        overall_status = "unsupported_risk"
    elif len(weakly_supported) > 0:
        overall_status = "partially_supported"
    else:
        overall_status = "supported"

    return {
        "overall_status": overall_status,
        "total_claims": total_claims,
        "supported_count": len(supported),
        "weakly_supported_count": len(weakly_supported),
        "unsupported_count": len(unsupported),
        "supported_claims": supported,
        "weakly_supported_claims": weakly_supported,
        "unsupported_claims": unsupported,
    }


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

    claims = [
        "RAG evaluation can include faithfulness, answer relevancy, context precision, and context recall.",
        "Citation quality is important when evaluating generated answers.",
        "All RAG systems use reinforcement learning as their main retrieval method.",
    ]

    verification_results = verify_answer_claims(claims, evidence)
    summary = detect_unsupported_claims(verification_results)

    print("\nUnsupported Claim Detection Summary")
    print("=" * 80)
    print(f"Overall status: {summary['overall_status']}")
    print(f"Total claims: {summary['total_claims']}")
    print(f"Supported: {summary['supported_count']}")
    print(f"Weakly supported: {summary['weakly_supported_count']}")
    print(f"Unsupported: {summary['unsupported_count']}")

    if summary["unsupported_claims"]:
        print("\nUnsupported Claims")
        print("-" * 80)
        for item in summary["unsupported_claims"]:
            print(f"- {item['claim']}")