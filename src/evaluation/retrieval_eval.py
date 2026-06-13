from pathlib import Path
import pandas as pd

from src.retrieval.dense_retriever import build_dense_index
from src.retrieval.bm25_retriever import build_bm25_index
from src.retrieval.hybrid_retriever import hybrid_search


TEST_QUERIES_PATH = Path("data/eval_set/retrieval_test_queries.csv")
OUTPUT_DIR = Path("experiments/retrieval_results")
OUTPUT_PATH = OUTPUT_DIR / "hybrid_retrieval_results.csv"


def evaluate_hybrid_retrieval(top_k: int = 5) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    queries_df = pd.read_csv(TEST_QUERIES_PATH)

    print("Building dense index...")
    build_dense_index()

    print("Building BM25 index...")
    bm25, bm25_chunks = build_bm25_index()

    rows = []

    for _, row in queries_df.iterrows():
        query_id = row["query_id"]
        question = row["question"]
        expected_papers = str(row["expected_relevant_papers"]).split(";")

        results = hybrid_search(
            query=question,
            bm25=bm25,
            bm25_chunks=bm25_chunks,
            top_k=top_k,
        )

        retrieved_papers = [result["paper_id"] for result in results]
        hit = any(paper in retrieved_papers for paper in expected_papers)

        for rank, result in enumerate(results, start=1):
            rows.append({
                "query_id": query_id,
                "question": question,
                "expected_relevant_papers": ";".join(expected_papers),
                "rank": rank,
                "retrieved_paper_id": result["paper_id"],
                "retrieved_chunk_id": result["chunk_id"],
                "hybrid_score": result["hybrid_score"],
                "dense_score": result["dense_score"],
                "bm25_score": result["bm25_score"],
                "hit_in_top_k": hit,
                "text_preview": result["text"][:300].replace("\n", " "),
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("Evaluation complete.")
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_hybrid_retrieval(top_k=5)