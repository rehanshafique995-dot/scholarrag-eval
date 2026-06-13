from src.retrieval.section_hybrid_retriever import (
    section_hybrid_search,
)
from src.retrieval.section_dense_retriever import build_section_dense_index
from src.retrieval.section_bm25_retriever import build_section_bm25_index


def format_evidence(results: list[dict]) -> str:
    """Format retrieved chunks as cited evidence."""
    evidence_blocks = []

    for i, result in enumerate(results, start=1):
        evidence_blocks.append(
            f"[Evidence {i}] "
            f"Paper: {result['paper_id']} | "
            f"Section: {result.get('section')} | "
            f"Chunk: {result.get('chunk_id')}\n"
            f"{result['text'][:900]}"
        )

    return "\n\n".join(evidence_blocks)


def generate_extractive_answer(query: str, results: list[dict]) -> str:
    """
    Generate a simple evidence-grounded answer from retrieved chunks.
    This is a baseline answer generator, not an LLM generator.
    """
    if not results:
        return "No relevant evidence was retrieved."

    top_papers = sorted(set(result["paper_id"] for result in results))
    top_sections = sorted(set(str(result.get("section")) for result in results))

    answer = f"""
Question:
{query}

Short Answer:
The retrieved evidence suggests that this question is mainly supported by papers {', '.join(top_papers)}. The most relevant sections include {', '.join(top_sections)}.

Evidence-Grounded Summary:
"""

    for i, result in enumerate(results[:3], start=1):
        preview = result["text"][:450].replace("\n", " ").strip()
        answer += (
            f"\n{i}. Based on {result['paper_id']} "
            f"({result.get('section')}, chunk {result.get('chunk_id')}), "
            f"the relevant evidence says: {preview}..."
        )

    answer += "\n\nCitations:\n"

    for i, result in enumerate(results[:5], start=1):
        answer += (
            f"- [{i}] {result['paper_id']} | "
            f"Section: {result.get('section')} | "
            f"Chunk: {result.get('chunk_id')}\n"
        )

    return answer.strip()


def answer_query(query: str, top_k: int = 5) -> str:
    """Retrieve evidence and generate an extractive answer."""
    build_section_dense_index()
    bm25, bm25_chunks = build_section_bm25_index()

    results = section_hybrid_search(
        query=query,
        bm25=bm25,
        bm25_chunks=bm25_chunks,
        top_k=top_k,
    )

    return generate_extractive_answer(query, results)


if __name__ == "__main__":
    test_query = "Which metrics are used to evaluate RAG systems?"
    final_answer = answer_query(test_query, top_k=5)

    print(final_answer)