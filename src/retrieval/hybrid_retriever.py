from dense_retriever import search as dense_search, build_dense_index
from bm25_retriever import build_bm25_index, search as bm25_search


def normalize_scores(results: list[dict]) -> list[dict]:
    """Normalize result scores between 0 and 1."""
    if not results:
        return results

    scores = [result["score"] for result in results]
    min_score = min(scores)
    max_score = max(scores)

    for result in results:
        if max_score == min_score:
            result["normalized_score"] = 1.0
        else:
            result["normalized_score"] = (
                result["score"] - min_score
            ) / (max_score - min_score)

    return results


def hybrid_search(
    query: str,
    bm25,
    bm25_chunks: list[dict],
    top_k: int = 5,
    dense_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """Combine dense and BM25 retrieval results."""
    dense_results = dense_search(query, top_k=top_k * 2)
    bm25_results = bm25_search(query, bm25, bm25_chunks, top_k=top_k * 2)

    dense_results = normalize_scores(dense_results)
    bm25_results = normalize_scores(bm25_results)

    combined = {}

    for result in dense_results:
        key = (result["paper_id"], result["chunk_id"])
        combined[key] = {
            **result,
            "dense_score": result["normalized_score"],
            "bm25_score": 0.0,
        }

    for result in bm25_results:
        key = (result["paper_id"], result["chunk_id"])

        if key not in combined:
            combined[key] = {
                **result,
                "dense_score": 0.0,
                "bm25_score": result["normalized_score"],
            }
        else:
            combined[key]["bm25_score"] = result["normalized_score"]

    final_results = []

    for item in combined.values():
        item["hybrid_score"] = (
            dense_weight * item["dense_score"]
            + bm25_weight * item["bm25_score"]
        )
        final_results.append(item)

    final_results = sorted(
        final_results,
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return final_results[:top_k]


if __name__ == "__main__":
    print("Building dense index...")
    build_dense_index()

    print("Building BM25 index...")
    bm25, bm25_chunks = build_bm25_index()

    query = "Which metrics are used to evaluate RAG systems?"

    results = hybrid_search(
        query=query,
        bm25=bm25,
        bm25_chunks=bm25_chunks,
        top_k=5
    )

    print(f"\nQUERY: {query}")
    print("=" * 100)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"Dense Score: {result['dense_score']:.4f}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Paper: {result['paper_id']} | Chunk: {result['chunk_id']}")
        print(result["text"][:700].replace("\n", " "))
        print("-" * 100)