
"""
PM-JAY 5,000-query RAG-only evaluation.

Pipeline:
    Local Ollama E5 embeddings
    -> Existing FAISS index
    -> Existing hybrid retrieval and ranking
    -> Top-5 unique package evaluation

Metrics:
    RAG Hit@1, Hit@3, Hit@5
    Recall@1, Recall@3, Recall@5
    Mean Reciprocal Rank (MRR)
    Per-query latency and execution status
    Wall-clock time and throughput

No Qwen reranking is used in this script.
"""

import csv
import os
import pickle
import time
from collections import Counter
from datetime import datetime

from config import TOP_K, INITIAL_RETRIEVAL_K, METADATA_PATH
from rag import retrieve


INPUT_FILE = "pmjay_final_evaluation_dataset_5000.csv"

# Separate files: never mix the previous Qwen results with RAG-only results.
OUTPUT_FILE = "pmjay_rag_only_results_5000.csv"
SUMMARY_FILE = "pmjay_rag_only_metrics_5000.txt"

# Set True to ignore previous RAG-only checkpoints and rerun all queries.
RERUN_ALL = False

# Validate expected package IDs against the knowledge-base metadata.
VALIDATE_GROUND_TRUTH = True

OUTPUT_FIELDS = [
    "query_id",
    "query",
    "expected_package_id",
    "difficulty",
    "query_type",

    "rag_result_1",
    "rag_score_1",
    "rag_result_2",
    "rag_score_2",
    "rag_result_3",
    "rag_score_3",
    "rag_result_4",
    "rag_score_4",
    "rag_result_5",
    "rag_score_5",

    "rag_hit_at_1",
    "rag_hit_at_3",
    "rag_hit_at_5",

    "rag_recall_at_1",
    "rag_recall_at_3",
    "rag_recall_at_5",

    "rag_mrr",
    "rag_status",
    "rag_error",
    "rag_latency_seconds",
    "total_query_seconds",
]


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def unique_package_candidates(candidates, limit):
    """Keep the highest-ranked candidate for each distinct package ID."""
    output = []
    seen = set()

    for candidate in candidates:
        package_id = str(candidate.get("package_id", "")).strip()

        if not package_id or package_id in seen:
            continue

        seen.add(package_id)
        output.append(candidate)

        if len(output) >= limit:
            break

    return output


def compute_rank_metrics(package_ids, expected):
    """Calculate Hit@1/3/5 and reciprocal rank."""
    ids = [
        str(package_id).strip()
        for package_id in package_ids
        if str(package_id).strip()
    ]
    expected = str(expected).strip()

    hit1 = int(bool(ids) and ids[0] == expected)
    hit3 = int(expected in ids[:3])
    hit5 = int(expected in ids[:5])

    mrr = 0.0

    for rank, package_id in enumerate(ids, start=1):
        if package_id == expected:
            mrr = 1.0 / rank
            break

    return hit1, hit3, hit5, mrr


def load_dataset():
    """Load and validate the input CSV."""
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE}. Put it beside this script "
            "or update INPUT_FILE."
        )

    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = set(reader.fieldnames or [])

    required = {"query_id", "query", "expected_package_id"}
    missing = required - fields

    if missing:
        raise ValueError(
            f"Input CSV is missing columns: {sorted(missing)}"
        )

    if not rows:
        raise ValueError("The input CSV has no data rows.")

    for line_number, row in enumerate(rows, start=2):
        for key in required:
            if not str(row.get(key, "")).strip():
                raise ValueError(
                    f"Blank {key!r} at input CSV line {line_number}."
                )

    query_ids = [str(row["query_id"]).strip() for row in rows]
    duplicates = [
        key for key, count in Counter(query_ids).items()
        if count > 1
    ]

    if duplicates:
        raise ValueError(
            f"Duplicate query_id values found: {duplicates[:10]}"
        )

    return rows


def validate_ground_truth(rows):
    """
    Verify expected package IDs appear in metadata.pkl.
    This checks ID membership, not clinical correctness of labels.
    """
    if not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}. "
            "Check the configured metadata path."
        )

    with open(METADATA_PATH, "rb") as file:
        metadata = pickle.load(file)

    if not isinstance(metadata, (list, tuple)):
        raise ValueError(
            "Expected metadata.pkl to contain a list or tuple of records."
        )

    known_ids = {
        str(record.get("package_id", "")).strip()
        for record in metadata
        if isinstance(record, dict)
        and str(record.get("package_id", "")).strip()
    }

    if not known_ids:
        raise ValueError("No package_id values found in metadata.pkl.")

    expected_ids = {
        str(row["expected_package_id"]).strip()
        for row in rows
    }

    unknown_ids = sorted(expected_ids - known_ids)

    print(f"Unique expected package IDs: {len(expected_ids)}")
    print(f"Package IDs found in metadata: {len(known_ids)}")

    if unknown_ids:
        preview = ", ".join(unknown_ids[:25])
        suffix = (
            f" (and {len(unknown_ids) - 25} more)"
            if len(unknown_ids) > 25 else ""
        )

        raise ValueError(
            "Ground-truth validation failed. These expected package IDs "
            f"were not found in metadata.pkl: {preview}{suffix}. "
            "Check package ID formatting and dataset labels."
        )

    print("Ground-truth validation passed.")


def load_completed():
    """Resume queries with successful RAG evaluation."""
    if RERUN_ALL or not os.path.isfile(OUTPUT_FILE):
        return {}

    completed = {}

    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            qid = str(row.get("query_id", "")).strip()

            if qid and row.get("rag_status") == "ok":
                completed[qid] = row

    return completed


def write_results(results_by_id, ordered_ids):
    """Atomically save the current checkpoint."""
    temporary = OUTPUT_FILE + ".tmp"

    with open(temporary, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=OUTPUT_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()

        for qid in ordered_ids:
            if qid in results_by_id:
                writer.writerow(results_by_id[qid])

    os.replace(temporary, OUTPUT_FILE)


def evaluate_query(row):
    qid = str(row["query_id"]).strip()
    query = str(row["query"]).strip()
    expected = str(row["expected_package_id"]).strip()

    started = time.perf_counter()
    rag_start = time.perf_counter()

    out = {field: "" for field in OUTPUT_FIELDS}
    out.update({
        "query_id": qid,
        "query": query,
        "expected_package_id": expected,
        "difficulty": row.get("difficulty", ""),
        "query_type": row.get("query_type", ""),
        "rag_status": "not_run",
        "rag_error": "",
        "rag_latency_seconds": 0.0,
        "total_query_seconds": 0.0,
    })

    candidates = []

    try:
        # Preserve the existing retrieval depth and hybrid ranking.
        raw_candidates = retrieve(
            query,
            top_k=max(TOP_K, INITIAL_RETRIEVAL_K),
        )

        candidates = unique_package_candidates(
            raw_candidates or [],
            TOP_K,
        )

        package_ids = [
            str(candidate.get("package_id", "")).strip()
            for candidate in candidates
        ]

        h1, h3, h5, mrr = compute_rank_metrics(
            package_ids,
            expected,
        )

        out.update({
            "rag_hit_at_1": h1,
            "rag_hit_at_3": h3,
            "rag_hit_at_5": h5,

            # With one expected relevant package per query,
            # these are equivalent to Hit@k.
            "rag_recall_at_1": h1,
            "rag_recall_at_3": h3,
            "rag_recall_at_5": h5,

            "rag_mrr": mrr,
            "rag_status": "ok",
        })

        for rank in range(1, TOP_K + 1):
            if rank <= len(candidates):
                out[f"rag_result_{rank}"] = (
                    candidates[rank - 1].get("package_id", "")
                )
                out[f"rag_score_{rank}"] = (
                    candidates[rank - 1].get("hybrid_score", "")
                )

    except Exception as exc:
        out.update({
            "rag_status": "error",
            "rag_error": f"{type(exc).__name__}: {exc}",
            "rag_hit_at_1": 0,
            "rag_hit_at_3": 0,
            "rag_hit_at_5": 0,
            "rag_recall_at_1": 0,
            "rag_recall_at_3": 0,
            "rag_recall_at_5": 0,
            "rag_mrr": 0.0,
        })

        for rank in range(1, TOP_K + 1):
            out[f"rag_result_{rank}"] = ""
            out[f"rag_score_{rank}"] = ""

    out["rag_latency_seconds"] = round(
        time.perf_counter() - rag_start, 6
    )
    out["total_query_seconds"] = round(
        time.perf_counter() - started, 6
    )

    return out


def average(rows, field):
    if not rows:
        return None

    return sum(
        as_float(row.get(field, 0))
        for row in rows
    ) / len(rows)


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def write_summary(rows, dataset_size, wall_seconds, run_started):
    rag_ok = [
        row for row in rows
        if row.get("rag_status") == "ok"
    ]

    rag_errors = sum(
        row.get("rag_status") == "error"
        for row in rows
    )

    no_candidates = sum(
        row.get("rag_status") == "ok"
        and not str(row.get("rag_result_1", "")).strip()
        for row in rows
    )

    throughput = (
        len(rows) / (wall_seconds / 60)
        if wall_seconds > 0 else None
    )

    lines = [
        "PM-JAY RAG-ONLY EVALUATION",
        "=" * 64,
        f"Run started: {run_started}",
        f"Summary written: {datetime.now().isoformat(timespec='seconds')}",
        f"Input dataset queries: {dataset_size}",
        f"Output rows: {len(rows)}",
        f"Queries without output rows: {max(0, dataset_size - len(rows))}",
        f"Current-run wall time (seconds): {wall_seconds:.3f}",
        f"Current-run wall time (minutes): {wall_seconds / 60:.3f}",
        f"Current-run wall time (hours): {wall_seconds / 3600:.3f}",
        f"Throughput (queries/minute): {fmt(throughput)}",
        "",
        "STAGE TIMING",
        (
            "RAG total seconds: "
            f"{sum(as_float(r.get('rag_latency_seconds')) for r in rows):.3f}"
        ),
        (
            "RAG average seconds per output row: "
            f"{fmt(average(rows, 'rag_latency_seconds'))}"
        ),
        (
            "Recorded total-query seconds: "
            f"{sum(as_float(r.get('total_query_seconds')) for r in rows):.3f}"
        ),
        "",
        "EXECUTION STATUS",
        f"RAG successful: {len(rag_ok)}",
        f"RAG errors: {rag_errors}",
        f"Successful queries with no candidates: {no_candidates}",
        "",
        "RAG METRICS (all output rows; errors count as zero)",
        f"Hit@1: {fmt(average(rows, 'rag_hit_at_1'))}",
        f"Hit@3: {fmt(average(rows, 'rag_hit_at_3'))}",
        f"Hit@5: {fmt(average(rows, 'rag_hit_at_5'))}",
        f"Recall@1: {fmt(average(rows, 'rag_recall_at_1'))}",
        f"Recall@3: {fmt(average(rows, 'rag_recall_at_3'))}",
        f"Recall@5: {fmt(average(rows, 'rag_recall_at_5'))}",
        f"MRR: {fmt(average(rows, 'rag_mrr'))}",
        "",
        "RAG METRICS (successful retrievals only)",
        f"Hit@1: {fmt(average(rag_ok, 'rag_hit_at_1'))}",
        f"Hit@3: {fmt(average(rag_ok, 'rag_hit_at_3'))}",
        f"Hit@5: {fmt(average(rag_ok, 'rag_hit_at_5'))}",
        f"Recall@1: {fmt(average(rag_ok, 'rag_recall_at_1'))}",
        f"Recall@3: {fmt(average(rag_ok, 'rag_recall_at_3'))}",
        f"Recall@5: {fmt(average(rag_ok, 'rag_recall_at_5'))}",
        f"MRR: {fmt(average(rag_ok, 'rag_mrr'))}",
        "",
        "NOTES",
        "- Qwen reranking is not used in this evaluation.",
        "- The existing retrieve() ranking determines the result order.",
        "- With one relevant package label per query, Recall@k equals Hit@k.",
        "- MRR uses the ordered, unique-package RAG results.",
        "- Current-run wall time excludes queries reused from checkpoints.",
        "- Metadata membership validation does not prove label correctness.",
    ]

    with open(SUMMARY_FILE, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nDetailed CSV: {OUTPUT_FILE}")
    print(f"Summary TXT:  {SUMMARY_FILE}")


def main():
    rows = load_dataset()

    if VALIDATE_GROUND_TRUTH:
        validate_ground_truth(rows)

    ordered_ids = [
        str(row["query_id"]).strip()
        for row in rows
    ]

    completed = load_completed()
    results_by_id = dict(completed)

    todo = [
        row for row in rows
        if str(row["query_id"]).strip() not in completed
    ]

    print("=" * 76)
    print("PM-JAY FINAL 5,000-QUERY RAG-ONLY EVALUATION")
    print("Local Ollama E5 + existing FAISS hybrid RAG")
    print("Qwen reranking: DISABLED")
    print("=" * 76)
    print(f"Dataset rows: {len(rows)}")
    print(f"Already completed and skipped: {len(completed)}")
    print(f"Queries to process: {len(todo)}")
    print(f"Results file: {OUTPUT_FILE}")
    print("Saving a checkpoint after every query.")

    run_started = datetime.now().isoformat(timespec="seconds")
    wall_start = time.perf_counter()

    try:
        for number, row in enumerate(todo, start=1):
            qid = str(row["query_id"]).strip()
            print(f"\n[{number}/{len(todo)}] {qid}: {row['query']}")

            result = evaluate_query(row)
            results_by_id[qid] = result

            print(
                f"RAG {result['rag_status']} | "
                f"Hit@1={result.get('rag_hit_at_1', '')} | "
                f"Hit@5={result.get('rag_hit_at_5', '')} | "
                f"RAG={result.get('rag_latency_seconds', 0)}s | "
                f"total={result.get('total_query_seconds', 0)}s"
            )

            if result.get("rag_error"):
                print("RAG ERROR:", result["rag_error"])

            # Save after every query for safe interruption and resumption.
            write_results(results_by_id, ordered_ids)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving completed query results...")
        write_results(results_by_id, ordered_ids)
        print(f"Checkpoint saved to: {OUTPUT_FILE}")
        print("Run this script again to resume.")
        return

    wall_seconds = time.perf_counter() - wall_start

    final_rows = [
        results_by_id[qid]
        for qid in ordered_ids
        if qid in results_by_id
    ]

    write_summary(
        final_rows,
        len(rows),
        wall_seconds,
        run_started,
    )


if __name__ == "__main__":
    main()
