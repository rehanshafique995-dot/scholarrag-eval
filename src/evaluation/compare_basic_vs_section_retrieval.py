from pathlib import Path
import pandas as pd

from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.bm25_retriever import build_bm25_index

from src.retrieval.section_hybrid_retriever import section_hybrid_search
from src.retrieval.section_dense_retriever import build_section_dense_index
from src.retrieval.section_bm25_retriever import build_section_bm25_index


TEST_QUERIES_PATH = Path("data/eval_set/retrieval_test_queries.csv")
OUTPUT_DIR = Path("experiments/retrieval_results")
OUTPUT_PATH = OUTPUT_DIR / "basic_vs_section_retrieval_comparison.csv"


def paper_hit(expected_papers: list[str], retrieved_papers: list[str]) -> bool:
    """Return True if at least one expected paper appears in retrieved papers."""
    return any(paper in retrieved_papers for paper in expected_papers)


def recall_at_k(expected_papers: list[str], retrieved_papers: list[str]) -> float:
    """Calculate paper-level recall@k."""
    expected_set = set(expected_papers)
    retrieved_set = set(retrieved_papers)

    if not expected_set:
        return 0.0

    return len(expected_set.intersection(retrieved_set)) / len(expected_set)


def evaluate_comparison(top_k: int = 5) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries_df = pd.read_csv(TEST_QUERIES_PATH)

    print("Building basic BM25 index...")
    basic_bm25, basic_chunks = build_bm25_index()

    print("Building section-aware dense index...")
    build_section_dense_index()

    print("Building section-aware BM25 index...")
    section_bm25, section_chunks = build_section_bm25_index()

    rows = []

    for _, row in queries_df.iterrows():
        query_id = row["query_id"]
        question = row["question"]
        query_type = row["query_type"]
        expected_papers = str(row["expected_relevant_papers"]).split(";")

        print(f"Evaluating {query_id}: {question}")

        basic_results = hybrid_search(
            query=question,
            bm25=basic_bm25,
            bm25_chunks=basic_chunks,
            top_k=top_k,
        )

        section_results = section_hybrid_search(
            query=question,
            bm25=section_bm25,
            bm25_chunks=section_chunks,
            top_k=top_k,
        )

        basic_papers = [result["paper_id"] for result in basic_results]
        section_papers = [result["paper_id"] for result in section_results]

        rows.append({
            "query_id": query_id,
            "question": question,
            "query_type": query_type,
            "expected_papers": ";".join(expected_papers),

            "basic_retrieved_papers": ";".join(basic_papers),
            "basic_hit_at_5": paper_hit(expected_papers, basic_papers),
            "basic_recall_at_5": round(recall_at_k(expected_papers, basic_papers), 3),

            "section_retrieved_papers": ";".join(section_papers),
            "section_hit_at_5": paper_hit(expected_papers, section_papers),
            "section_recall_at_5": round(recall_at_k(expected_papers, section_papers), 3),

            "basic_top_result": basic_results[0]["text"][:250].replace("\n", " ") if basic_results else "",
            "section_top_result": section_results[0]["text"][:250].replace("\n", " ") if section_results else "",
            "section_top_section": section_results[0].get("section") if section_results else "",
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUTPUT_PATH, index=False)

    print("\nComparison complete.")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nOverall Summary")
    print("=" * 60)
    print(f"Basic Mean Hit@5: {comparison_df['basic_hit_at_5'].mean():.3f}")
    print(f"Basic Mean Recall@5: {comparison_df['basic_recall_at_5'].mean():.3f}")
    print(f"Section Mean Hit@5: {comparison_df['section_hit_at_5'].mean():.3f}")
    print(f"Section Mean Recall@5: {comparison_df['section_recall_at_5'].mean():.3f}")


if __name__ == "__main__":
    evaluate_comparison(top_k=5)