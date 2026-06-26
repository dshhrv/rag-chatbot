# HSE Regulation RAG Chatbot

<p align="center">
  <b>Citation-grounded assistant for HSE students and applicants</b><br/>
  Answers questions about university regulations using retrieved document fragments.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/RAG-Hybrid%20Retrieval-6A5ACD" alt="RAG">
  <img src="https://img.shields.io/badge/Qdrant-Vector%20Search-DC244C?logo=qdrant&logoColor=white" alt="Qdrant">
  <img src="https://img.shields.io/badge/LLM-Local%20GGUF-111111" alt="Local LLM">
  <img src="https://img.shields.io/badge/Monitoring-Prometheus%20%2B%20Grafana-E6522C?logo=prometheus&logoColor=white" alt="Monitoring">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#pipeline">Pipeline</a> ·
  <a href="#results">Results</a> ·
  <a href="#tech-stack">Tech stack</a> ·
  <a href="#project-status">Project status</a>
</p>

---

## Overview

This repository contains a Retrieval-Augmented Generation (RAG) chatbot for HSE students and applicants. The system answers questions based on the university regulation **"Popatkus"** and returns citations to the exact document clauses used in the answer.

The project is designed for routine questions about academic procedures and rules. Its main principle is simple: when the system cannot find direct evidence in the document, it does not invent an answer. Instead, it returns a safe response and recommends contacting support when necessary.

### What the system does

- retrieves relevant fragments from the regulation in Russian and English
- combines semantic search with exact-term search
- reranks retrieved passages before generation
- generates concise structured answers with citations
- provides glossary definitions without calling the LLM when possible
- routes ambiguous, unsafe, or unsupported requests to clarification, refusal, or escalation
- monitors latency, errors, and potential hallucination blocks

---

## Pipeline

<p align="center">
  <img src="./assets/architecture.svg" alt="RAG system architecture" width="900">
</p>

```text
User question
    ↓
Web UI → Python backend → Agentic routing
    ↓
Dense retrieval (Qdrant) + sparse retrieval (BM25)
    ↓
Reciprocal Rank Fusion → cross-encoder reranking
    ↓
Structured generation with a local GGUF model via llama.cpp
    ↓
Citation validation + selective NLI verification
    ↓
Answer with citations / safe refusal / escalation
```

### Knowledge base

The source regulation is parsed from DOCX, cleaned, and split into semantic chunks. Each chunk is stored together with metadata used for retrieval and citations:

```text
chunk_id · type · lang · clause_id · heading_path
```

Tables, glossary terms, notes, and regular clauses are handled separately. This makes exact definitions and abbreviations easier to retrieve and cite.

### Retrieval

The search stage uses a hybrid retrieval pipeline:

- **Dense retrieval** via Qdrant with `intfloat/multilingual-e5-base`
- **BM25** for exact wording, terms, and abbreviations such as ИУП or КУД
- **Reciprocal Rank Fusion** to combine dense and sparse candidates
- **Cross-encoder reranking** to select the most relevant context for generation

### Generation and verification

Answers are generated locally with a GGUF model through `llama.cpp`. The model is constrained to return structured output:

```json
{
  "answer": "…",
  "citations": ["…"],
  "found": true
}
```

The system validates returned citation ids against retrieved context. If an answer has no citations, a selective NLI verifier can be used as an additional signal. When evidence is missing or the request requires an individual decision, the chatbot returns a safe refusal or escalation instead of guessing.

---

## Results

Evaluation is performed on a manually prepared golden set of regulation-related questions.

| Metric    |      Score |
| --------- | ---------: |
| Recall@10 | **0.9796** |
| Hit@10    | **0.9917** |

Besides retrieval quality, the project tracks citation coverage, valid citation rate, structured-output validity, latency by pipeline stage, escalation rate, errors, and blocked potential hallucinations.

---

## Tech stack

| Area              | Tools                              |
| ----------------- | ---------------------------------- |
| Backend           | Python                             |
| Web interface     | Web UI                             |
| Vector search     | Qdrant                             |
| Sparse retrieval  | BM25                               |
| Embeddings        | `intfloat/multilingual-e5-base`    |
| Reranking         | Cross-encoder                      |
| Local generation  | GGUF model + `llama.cpp`           |
| Structured output | SGR + grammar-constrained decoding |
| Request tracing   | Langfuse                           |
| Monitoring        | Prometheus + Grafana               |
| Deployment        | Docker + Yandex Cloud              |

---

## Key design decisions

### Ground answers in the source document

The LLM receives retrieved regulation fragments rather than answering from its internal knowledge alone. Every final answer is expected to contain citations to the supporting clauses.

### Prioritize retrieval and verification over fine-tuning

The project focuses on document parsing, chunking, hybrid retrieval, reranking, structured generation, and citation validation. Fine-tuning is not used in the current version.

### Use local inference

Generation is performed with a local GGUF model instead of external LLM APIs. This keeps the inference path self-hosted and makes the deployment independent of external model providers.

### Fail safely

The system does not answer questions outside the document, personal cases requiring a human decision, or requests without source support. These cases are routed to a safe response or escalation.

---

## Project status

This repository contains a research implementation of a citation-grounded RAG chatbot.

The full local deployment requires separate setup of the GGUF model, Qdrant, monitoring services, environment variables, and prepared document indexes. The project is not currently packaged as a one-command installation.

---

## Repository contents

```text
assets/                  architecture diagram and visual materials
data/                    prepared document chunks and evaluation data
docker-compose*.yml      infrastructure and monitoring services
prometheus.yml           Prometheus configuration
src/ / app/              backend and RAG pipeline modules
```

> The exact directory structure may vary between branches and local environments.

---

## Future work

- package reproducible local deployment
- add automated document-change detection and reindexing
- improve multi-turn dialogue handling
- add optional user feedback collection
- move local generation to GPU inference for lower latency
