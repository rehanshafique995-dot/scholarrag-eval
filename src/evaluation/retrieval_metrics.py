from pathlib import Path
import pandas as pd


INPUT_PATH = Path("experiments/retrieval_results/hybrid_retrieval_results.csv")
OUTPUT_PATH = Path("experiments/retrieval_results/hybrid_retrieval_summary.csv")


def calculate_metrics() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "hybrid_retrieval_results.csv not found. Run retrieval_eval.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    summary_rows = []

    for query_id, group in df.groupby("query_id"):
        question = group["question"].iloc[0]
        expected_papers = set(str(group["expected_relevant_papers"].iloc[0]).split(";"))
        retrieved_papers = set(group["retrieved_paper_id"].astype(str).tolist())

        matched_papers = expected_papers.intersection(retrieved_papers)

        hit_at_5 = len(matched_papers) > 0
        recall_at_5 = len(matched_papers) / len(expected_papers) if expected_papers else 0

        avg_hybrid_score = group["hybrid_score"].mean()

        summary_rows.append({
            "query_id": query_id,
            "question": question,
            "expected_papers": ";".join(sorted(expected_papers)),
            "retrieved_papers": ";".join(sorted(retrieved_papers)),
            "matched_papers": ";".join(sorted(matched_papers)),
            "hit_at_5": hit_at_5,
            "recall_at_5": round(recall_at_5, 3),
            "avg_hybrid_score": round(avg_hybrid_score, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_PATH, index=False)

    print("Retrieval metrics summary created.")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nOverall Metrics")
    print("=" * 50)
    print(f"Mean Hit@5: {summary_df['hit_at_5'].mean():.3f}")
    print(f"Mean Recall@5: {summary_df['recall_at_5'].mean():.3f}")


if __name__ == "__main__":
    calculate_metrics()
    