# ScholarRAG-Eval

**ScholarRAG-Eval** is an independent research-engineering project focused on building a citation-grounded Global RAG system for academic literature review, corpus-level reasoning, hallucination detection, and RAG evaluation.

This is not a formal final-year project. It is a self-directed experimental research project developed to build deeper practical and research exposure in Retrieval-Augmented Generation, LLM evaluation, trustworthy AI systems, and academic literature analysis. The project is being developed with informal academic supervision and guidance.

---

## Project Vision

The goal of this project is to move beyond a basic PDF chatbot and build a research-oriented RAG system that can:

* Ingest academic research papers
* Parse and structure paper content
* Retrieve evidence using dense, BM25, and hybrid retrieval
* Answer local, comparative, and corpus-level literature-review questions
* Generate citation-grounded answers
* Verify whether claims are supported by cited evidence
* Detect unsupported or hallucinated claims
* Evaluate retrieval and generation quality using measurable metrics

---

## Problem Statement

Most beginner RAG systems focus on simple document question answering:

```text
Upload PDF → Retrieve chunks → Generate answer
```

However, academic literature review requires more than local document search. A useful research assistant should be able to compare multiple papers, identify recurring methods, extract common datasets and metrics, detect repeated limitations, and verify whether generated claims are actually supported by evidence.

This project explores how RAG systems can support academic literature review more reliably through corpus-level reasoning, citation verification, and hallucination evaluation.

---

## Core Research Direction

This project focuses on four main areas:

1. **Academic Paper RAG**
2. **Global / Corpus-Level Reasoning**
3. **Citation Verification**
4. **Hallucination / Unsupported-Claim Detection**

---

## Planned MVP Scope

The first version will include:

* 30 academic papers
* Local, comparative, and global question types
* PDF parsing and section-aware chunking
* Dense retrieval baseline
* BM25 retrieval baseline
* Hybrid retrieval
* Baseline answer generation
* Citation support checking
* Unsupported-claim detection
* RAG evaluation metrics
* Simple Streamlit demo interface

---

## Current Status

**Stage:** Day 01 — Repository setup and project foundation.

The initial focus is to create a clean project structure, document the research direction, and prepare the foundation for corpus collection and PDF parsing.

---

## Technology Stack

Planned first-version stack:

* Python
* Jupyter Notebook
* Streamlit
* PyMuPDF
* pdfplumber
* pandas
* numpy
* sentence-transformers
* FAISS
* rank-bm25
* RAGAS
* Git + GitHub

Advanced tools such as Qdrant, FastAPI, Docker, GraphRAG, agentic workflows, and multilingual RAG may be considered later as future extensions.

---

## Repository Structure

```text
scholarrag-eval/
├── README.md
├── proposal/
├── papers/
│   ├── pdfs/
│   ├── parsed/
│   └── metadata.csv
├── data/
│   ├── chunks/
│   ├── indexes/
│   ├── eval_set/
│   └── structured/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── reasoning/
│   ├── evaluation/
│   └── utils/
├── app/
├── experiments/
├── reports/
├── configs/
├── tests/
├── requirements.txt
└── LICENSE
```

---

## Initial Milestones

* [ ] Create repository and project structure
* [ ] Prepare metadata schema
* [ ] Collect first 10 academic papers
* [ ] Test PDF parsing
* [ ] Implement fixed-size chunking
* [ ] Implement section-aware chunking
* [ ] Build dense retrieval baseline
* [ ] Build BM25 retrieval baseline
* [ ] Compare dense vs BM25 retrieval
* [ ] Build hybrid retrieval
* [ ] Create first evaluation question set

---

## Long-Term Goal

The long-term goal is to develop this into a strong research portfolio project that demonstrates practical ability in AI systems, NLP, RAG pipelines, retrieval evaluation, citation-grounded generation, and trustworthy LLM-based academic tools.

This project is intended to support learning, experimentation, research maturity, GitHub portfolio development, and future graduate research preparation.

---

## Disclaimer

This repository is part of an independent learning and research-development effort. It is not an official final-year project submission. Research papers used for experimentation will be handled responsibly, and raw copyrighted PDFs may not be uploaded publicly. Metadata, code, documentation, and reproducible experiment logic will be shared where appropriate.

---

## License

This project is planned to be released under the MIT License.
