# AI-Assisted Claim Adjudication for Government Healthcare Schemes: A PM-JAY Case Study

## Overview

This project explores the application of **Retrieval-Augmented Generation (RAG)** techniques to healthcare benefit-package identification under India's Pradhan Mantri Jan Arogya Yojana (PM-JAY). It focuses on improving the retrieval and ranking of relevant healthcare packages from a structured knowledge base using natural-language queries.

The system implements a **weighted hybrid retrieval approach** that combines semantic similarity, lexical relevance, package-specific matching, treatment-identity matching, and treatment-variant matching. It uses locally executed E5 embeddings and FAISS-based vector search to retrieve and rank relevant PM-JAY package records.

The final evaluation focuses exclusively on the weighted hybrid RAG pipeline, without an additional language-model reranking stage.

## Problem Statement

Identifying the appropriate healthcare benefit package from a large catalogue can be challenging when queries contain alternative terminology, medical abbreviations, or descriptions that differ from the wording in the source records.

Pure keyword-based retrieval may miss relevant records when terminology differs, while semantic similarity alone may return closely related but incorrect procedures.

This project addresses these challenges by integrating complementary retrieval signals into a domain-aware ranking strategy designed for PM-JAY package identification.

## Objectives

- Develop a retrieval pipeline for natural-language PM-JAY package queries.
- Combine semantic and lexical relevance with package- and treatment-specific matching.
- Improve the ranking of relevant healthcare benefit packages.
- Evaluate retrieval performance using Hit@k, Recall@k, and Mean Reciprocal Rank.
- Measure per-query latency and monitor execution status.
- Investigate the feasibility of weighted hybrid retrieval as a supporting component in healthcare claims information systems.

## System Architecture

```text
Natural-Language Query
          |
          v
Local E5 Embedding Model
          |
          v
Existing FAISS Vector Index
          |
          v
Candidate Retrieval
          |
          v
Weighted Hybrid Ranking
          |
          v
Unique Package Selection
          |
          v
Top-5 Ranked Packages
          |
          v
Evaluation Against Ground Truth
```

## Technologies Used

| Component | Technology |
|---|---|
| Programming language | Python |
| Embedding model | `jeffh/intfloat-multilingual-e5-large:q8_0` |
| Local model runtime | Ollama |
| Vector search | FAISS |
| Retrieval approach | Weighted hybrid retrieval |
| Evaluation data | Custom PM-JAY query dataset |
| Evaluation metrics | Hit@k, Recall@k, MRR |
| Performance measurement | Python `time` module |
| Data processing | CSV, Python collections, Pickle |

## Weighted Hybrid Retrieval

The ranking mechanism combines multiple relevance signals to identify the most appropriate package for a given query.

The configured hybrid scoring function is:

\[
\begin{aligned}
S(q,p) ={}& 0.45S_{\mathrm{semantic}}\\
&+0.20S_{\mathrm{lexical}}\\
&+0.10S_{\mathrm{package}}\\
&+0.20S_{\mathrm{treatment}}\\
&+0.05S_{\mathrm{variant}}
\end{aligned}
\]

Here, \(q\) represents the query and \(p\) represents a candidate package record.

The semantic component measures the relevance of the query and package representation. The lexical component captures term-level matching. Package-specific matching contributes evidence associated with package identifiers or package-related terms. Treatment-identity matching helps distinguish procedures, while treatment-variant matching captures relevant distinctions between treatment variations.

The implementation also applies a penalty to certain hard treatment-identity mismatches. This is intended to reduce the ranking of candidates that are semantically related but refer to a different treatment identity.

The weights and matching rules are implementation-specific configuration choices rather than universal optimal values.

## Evaluation Dataset

The evaluation uses a custom dataset containing natural-language queries paired with expected PM-JAY package identifiers. Each record includes a query identifier, query text, expected package identifier, difficulty label, and query-type label.

The dataset was planned to contain 5,000 queries. However, the final recorded RAG-only evaluation contains **4,431 evaluated queries**. All results reported below refer to these evaluated records, not the entire planned dataset.

The evaluation includes three difficulty categories: basic, easy, and medium. It also contains nine query-type categories, labelled `template_1` through `template_9`, allowing performance to be examined across different query formulations.

## Evaluation Methodology

For each query, the system generates a local E5 embedding and retrieves candidates from the existing FAISS index. The weighted hybrid ranking mechanism orders the candidates, after which duplicate package identifiers are removed while retaining their highest-ranked occurrences.

The first five unique package candidates are compared against the expected package identifier. The evaluation records ranking metrics, package identifiers, hybrid scores, execution status, and per-query timing.

### Metrics

- **Hit@1:** Proportion of queries for which the expected package is ranked first.
- **Hit@3:** Proportion of queries for which the expected package appears within the first three candidates.
- **Hit@5:** Proportion of queries for which the expected package appears within the first five candidates.
- **Recall@k:** In this single-relevant-package evaluation, Recall@k is equivalent to Hit@k.
- **Mean Reciprocal Rank (MRR):** Measures how highly the expected package appears in the ranked results.
- **Query latency:** Measures the recorded retrieval processing time for each query.

## Experimental Results

The following results are from the final RAG-only evaluation CSV containing 4,431 queries.

| Metric | Result |
|---|---:|
| Evaluated queries | 4,431 |
| Hit@1 | 94.15% |
| Hit@3 | 99.01% |
| Hit@5 | 99.41% |
| Recall@1 | 94.15% |
| Recall@3 | 99.01% |
| Recall@5 | 99.41% |
| Mean Reciprocal Rank | 0.9658 |
| RAG execution errors recorded | 0 |
| Queries where the expected package was absent from the top five | 26 |
| Mean recorded query latency | Approximately 0.181 seconds |
| Median recorded query latency | Approximately 0.175 seconds |

### Results Interpretation

The system achieved a Hit@1 of 94.15%, showing that the expected package was ranked first for most evaluated queries. Hit@3 increased to 99.01%, while Hit@5 reached 99.41%. This indicates that the expected package was available within the first five unique candidates for nearly all evaluated queries.

The MRR of 0.9658 further indicates that the expected package generally appeared near the top of the ranked list. The results suggest that the weighted hybrid retrieval mechanism is effective for the evaluated PM-JAY package-identification task.

The 26 queries for which the expected package was absent from the top five provide opportunities for further error analysis. Future work could examine similar procedure descriptions, treatment-specific distinctions, terminology variation, and ranking-weight sensitivity.

The recorded mean query latency was approximately 0.181 seconds. This measurement describes the retrieval latency recorded in the evaluation file; it is not a benchmark against other systems or a measurement of the complete end-to-end claim-adjudication process.

## Repository Structure

The following is a representative structure for the project. Adjust filenames to match the files included in your actual repository.

```text
pmjay-hybrid-rag/
├── config.py
├── rag.py
├── ingest.py
├── evaluate_pmjay_rag_only.py
├── pmjay_final_evaluation_dataset_5000.csv
├── Health_benefit_Packages_1_0_a2f1170337_RAG_CLEAN_FULL.txt
├── vector_db/
│   ├── pmjay.index
│   └── metadata.pkl
├── pmjay_rag_only_results_5000.csv
├── pmjay_rag_only_metrics_5000.txt
├── requirements.txt
└── README.md
```

The evaluation script and generated result filenames should match the versions used in your final experiment. Large model files, datasets, indexes, and generated results may be excluded from GitHub when repository size or data-sharing restrictions apply.

## Setup and Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_REPOSITORY_DIRECTORY
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the dependencies required by the project modules and verify their versions against the working environment.

### 4. Install and verify Ollama

Install Ollama and ensure the configured embedding model is available locally.

```powershell
ollama pull jeffh/intfloat-multilingual-e5-large:q8_0
ollama list
```

The project configuration must reference the same embedding model used to create the existing FAISS index. Changing the embedding model may require rebuilding the index.

### 5. Verify project files

Ensure that the configuration, retrieval module, evaluation dataset, FAISS index, and corresponding metadata file are present at the paths specified in `config.py`.

## Running the Evaluation

Run the RAG-only evaluation script from the project root:

```powershell
python evaluate_pmjay_rag_only.py
```

Replace the script filename with the actual name of your final RAG-only evaluation script if it differs.

The evaluation script validates required dataset fields and expected package identifiers, retrieves and ranks package candidates, calculates the evaluation metrics, and saves checkpoints after each processed query. If interrupted, it can resume from successfully completed queries when the existing output file and checkpoint settings are preserved.

The results are saved to the configured RAG-only CSV file, while the aggregate metrics and timing summary are written to a text report. The exact filenames are determined by the constants in the evaluation script.

## Research Contribution

The project investigates a PM-JAY-specific weighted hybrid retrieval strategy that integrates semantic similarity with lexical and treatment-aware relevance signals. Its contribution lies in the design and empirical evaluation of these complementary signals for healthcare benefit-package identification, together with a multi-query evaluation covering different difficulty and query-type categories.

The reported results provide evidence of retrieval effectiveness on the evaluated dataset. They do not independently establish the correctness of healthcare claim approvals or rejections, clinical decisions, patient eligibility determinations, or financial settlements.

## Limitations and Future Work

The reported results are based on 4,431 evaluated queries rather than the full planned dataset of 5,000 queries. Evaluation quality also depends on the correctness of the expected package identifiers and the representativeness of the query dataset.

Future work may include detailed error analysis, validation of ground-truth labels, testing on independently collected queries, examination of ranking-weight sensitivity, and evaluation across additional healthcare package catalogues. The system could also be investigated as a retrieval-support component within a broader claim-adjudication workflow, subject to appropriate domain validation and human oversight.

## Disclaimer

This project is an academic research prototype focused on retrieving PM-JAY healthcare benefit-package information. It is not a certified healthcare claims adjudication system and should not be used as the sole basis for clinical, eligibility, reimbursement, or claim-approval decisions.

## Acknowledgements

This work was developed as an academic project investigating the application of retrieval-based AI techniques to government healthcare benefit-package information.
