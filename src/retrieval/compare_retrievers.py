from dense_retriever import search as dense_search, build_dense_index
from bm25_retriever import build_bm25_index, search as bm25_search


TEST_QUERIES = [
    "What is retrieval augmented generation?",
    "Which metrics are used to evaluate RAG systems?",
    "What is citation verification in LLM generated answers?",
    "How does GraphRAG answer global questions?",
    "What problem does lost in the middle describe?"
]


def print_results(title: str, results: list[dict]) -> None:
    print(f"\n{title}")
    print("=" * 100)

    for i, result in enumerate(results, start=1):
        print(f"\nRank {i}")
        print(f"Score: {result['score']}")
        print(f"Paper: {result['paper_id']} | Chunk: {result['chunk_id']}")
        print(result["text"][:500].replace("\n", " "))
        print("-" * 100)


def main() -> None:
    print("Building/loading dense index...")
    build_dense_index()

    print("Building BM25 index...")
    bm25, chunks = build_bm25_index()

    for query in TEST_QUERIES:
        print("\n\n" + "#" * 120)
        print(f"QUERY: {query}")
        print("#" * 120)

        dense_results = dense_search(query, top_k=3)
        bm25_results = bm25_search(query, bm25, chunks, top_k=3)

        print_results("DENSE RESULTS", dense_results)
        print_results("BM25 RESULTS", bm25_results)


if __name__ == "__main__":
    main()