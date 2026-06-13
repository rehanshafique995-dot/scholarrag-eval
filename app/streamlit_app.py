import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.reasoning.rag_pipeline import run_rag_pipeline


st.set_page_config(
    page_title="ScholarRAG-Eval",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ScholarRAG-Eval")
st.subheader("Citation-Grounded RAG for Academic Literature Review")

st.markdown(
    """
    This demo retrieves evidence from a 10-paper RAG research corpus,
    generates an evidence-based answer, verifies claims against retrieved chunks,
    and flags unsupported claims.
    """
)

query = st.text_input(
    "Enter your research question:",
    value="Which metrics are used to evaluate RAG systems?",
)

top_k = st.slider(
    "Number of evidence chunks to retrieve:",
    min_value=3,
    max_value=10,
    value=5,
)

run_button = st.button("Run ScholarRAG-Eval")

if run_button:
    if not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Running retrieval, answer generation, and citation verification..."):
            result = run_rag_pipeline(query=query, top_k=top_k)

        st.success("Pipeline completed.")

        st.header("Generated Answer")
        st.write(result["answer"])

        st.header("Support Summary")
        summary = result["support_summary"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Status", summary["overall_status"])
        col2.metric("Total Claims", summary["total_claims"])
        col3.metric("Supported", summary["supported_count"])
        col4.metric("Unsupported", summary["unsupported_count"])

        st.header("Retrieved Evidence")

        for i, chunk in enumerate(result["evidence"], start=1):
            with st.expander(
                f"Evidence {i}: {chunk['paper_id']} | {chunk.get('section')} | Chunk {chunk.get('chunk_id')}"
            ):
                st.write(chunk["text"])

        st.header("Claim Verification")

        for item in result["verification_results"]:
            status = item["support_status"]

            if status == "supported":
                st.success(f"SUPPORTED: {item['claim']}")
            elif status == "weakly_supported":
                st.warning(f"WEAKLY SUPPORTED: {item['claim']}")
            else:
                st.error(f"UNSUPPORTED: {item['claim']}")

            st.caption(
                f"Best evidence: {item['best_paper_id']} | "
                f"{item['best_section']} | Chunk {item['best_chunk_id']} | "
                f"Score: {item['best_score']}"
            )