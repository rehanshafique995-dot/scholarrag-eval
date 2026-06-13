# Week 01 Progress Log

## Project
ScholarRAG-Eval: Citation-Grounded Global RAG System for Academic Literature Review and Hallucination Verification

## Project Type
Independent research-engineering project, not a formal final-year project.

## Week Focus
Repository setup, corpus metadata creation, PDF ingestion pipeline, and retrieval baseline development.

## Work Completed

- Created GitHub repository
- Added README.md
- Added Python .gitignore
- Added project directory structure
- Created requirements.txt
- Created paper metadata schema
- Added metadata for first 10 RAG-related papers
- Downloaded PDFs locally
- Built PDF text extraction script
- Parsed PDFs into raw text
- Built text cleaning script
- Built section detection script
- Built text chunking script
- Built dense retrieval baseline using sentence-transformers and FAISS
- Built BM25 sparse retrieval baseline
- Built hybrid retrieval baseline
- Created initial retrieval test queries
- Built hybrid retrieval evaluation script
- Built retrieval metrics summary script

## Current Corpus Status

| Item | Count |
|---|---:|
| Papers in metadata | 10 |
| PDFs collected locally | 10 |
| Parsed text files | 10 |
| Retrieval test queries | 10 |

## Current Technical Status

The project currently supports PDF parsing, text cleaning, chunking, dense retrieval, BM25 retrieval, hybrid retrieval, and basic retrieval evaluation.

## Important Learning

This week clarified that the project is not just a PDF chatbot. It is a research-engineering system focused on retrieval quality, citation grounding, corpus-level reasoning, and hallucination evaluation.

## Problems Faced

- Git remote/local conflict due to separate initialization on GitHub and local machine
- Merge conflicts in README.md and .gitignore
- PyMuPDF import issue due to package/environment mismatch
- Need to keep raw PDFs and parsed text local instead of pushing them to GitHub

## Solutions Applied

- Resolved unrelated Git histories
- Fixed merge conflicts
- Activated virtual environment correctly
- Installed requirements properly
- Updated .gitignore to exclude PDFs, parsed files, cleaned files, chunks, indexes, and experiment outputs

## Next Week Plan

- Improve chunk metadata
- Add section-aware chunking
- Improve retrieval evaluation metrics
- Create local/comparative/global question categories
- Begin answer generation module
- Start citation-grounded response design