
import re
import pickle

import faiss
import numpy as np
import ollama

from config import (
    EMBEDDING_MODEL,
    INDEX_PATH,
    METADATA_PATH,
    TOP_K,
    INITIAL_RETRIEVAL_K,
)


# ============================================================
# LOAD DATABASE
# ============================================================

print("\nLoading FAISS vector database...")

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

if index.ntotal != len(metadata):
    raise ValueError(
        f"FAISS index contains {index.ntotal} vectors, "
        f"but metadata contains {len(metadata)} records."
    )

print(f"Loaded {index.ntotal} FAISS vectors.")
print(f"Loaded {len(metadata)} metadata records.")
print(f"Embedding model: {EMBEDDING_MODEL}")


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(text).lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what", "is", "the", "a", "an", "for", "of", "in",
    "on", "to", "and", "are", "was", "were", "required",
    "requirements", "how", "much", "does", "do", "i",
    "need", "tell", "me", "about", "can", "you", "give",
    "show", "please", "package", "rate", "cost", "price",
}


# ============================================================
# TREATMENT IDENTITY
# ============================================================

TREATMENT_IDENTITIES = {

    "cobalt_60": [
        "cobalt 60",
        "cobalt60",
        "cobalt",
    ],

    "linear_accelerator": [
        "linear accelerator",
        "linear accelerator external beam",
    ],

    "electron_beam": [
        "electron beam",
    ],

    "brachytherapy": [
        "brachytherapy",
    ],

    "tomotherapy": [
        "tomotherapy",
    ],

    "gamma_knife": [
        "gamma knife",
    ],

    "cyberknife": [
        "cyberknife",
        "cyber knife",
    ],
}


# ============================================================
# TREATMENT VARIANTS
# ============================================================

VARIANT_PHRASES = {

    "palliative": [
        "palliative",
    ],

    "radical": [
        "radical",
    ],

    "adjuvant": [
        "adjuvant",
    ],

    "neoadjuvant": [
        "neoadjuvant",
    ],

    "imrt": [
        "imrt",
        "intensity modulated radiotherapy",
    ],

    "igrt": [
        "igrt",
        "image guided radiotherapy",
    ],

    "srt": [
        "srt",
        "stereotactic radiotherapy",
    ],

    "srs": [
        "srs",
        "stereotactic radiosurgery",
    ],

    "3d_crt": [
        "3d crt",
        "3d crt 2d planning",
        "2d planning",
    ],
}


# ============================================================
# DETECT TREATMENT IDENTITY
# ============================================================

def detect_treatment_identity(text):

    text = normalize_text(text)
    detected = []

    for identity, phrases in TREATMENT_IDENTITIES.items():

        for phrase in phrases:

            phrase_normalized = normalize_text(phrase)

            if phrase_normalized in text:
                detected.append(identity)
                break

    return list(set(detected))


# ============================================================
# DETECT VARIANTS
# ============================================================

def detect_variants(text):

    text = normalize_text(text)
    detected = []

    for variant, phrases in VARIANT_PHRASES.items():

        for phrase in phrases:

            phrase_normalized = normalize_text(phrase)

            if phrase_normalized in text:
                detected.append(variant)
                break

    return list(set(detected))


# ============================================================
# CLEAN QUERY
# ============================================================

def clean_query(query):

    words = normalize_text(query).split()

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)


# ============================================================
# OLLAMA QUERY EMBEDDING
# ============================================================

def embed_query(query):

    """
    Generate a normalized query embedding using Ollama.

    The query prefix must match the E5 embedding convention
    used when building the FAISS database.
    """

    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input="query: " + query,
        )

    except Exception as error:
        raise RuntimeError(
            f"Ollama query embedding failed. "
            f"Check that Ollama is running and model "
            f"'{EMBEDDING_MODEL}' is available. "
            f"Original error: {error}"
        ) from error

    embeddings = response.get("embeddings")

    if embeddings is None and hasattr(response, "embeddings"):
        embeddings = response.embeddings

    if not embeddings:
        raise ValueError(
            "Ollama returned an empty query embedding."
        )

    vector = np.asarray(
        embeddings[0],
        dtype=np.float32,
    ).reshape(1, -1)

    if vector.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension mismatch: query vector has "
            f"{vector.shape[1]} dimensions, but FAISS expects "
            f"{index.d}. Rebuild the index using the same "
            f"Ollama embedding model."
        )

    norm = np.linalg.norm(
        vector,
        axis=1,
        keepdims=True,
    )

    norm = np.maximum(norm, 1e-12)

    return np.ascontiguousarray(vector / norm)


# ============================================================
# TREATMENT IDENTITY SCORE
# ============================================================

def treatment_identity_score(query, record):

    query_identities = detect_treatment_identity(query)
    record_identities = detect_treatment_identity(record)

    # Query does not specify a treatment identity.
    if not query_identities:
        return 0.0

    # At least one requested treatment identity matches.
    for identity in query_identities:

        if identity in record_identities:
            return 1.0

    # Explicit treatment identity mismatch.
    return -1.0


# ============================================================
# VARIANT SCORE
# ============================================================

def variant_score(query, record):

    query_variants = detect_variants(query)
    record_variants = detect_variants(record)

    # No variant requested.
    if not query_variants:
        return 0.0

    matched = sum(
        1
        for variant in query_variants
        if variant in record_variants
    )

    # All requested variants match.
    if matched == len(query_variants):
        return 1.0

    # Partial variant match.
    if matched > 0:
        return 0.5

    # Explicit variant mismatch.
    return -1.0


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_score(query, package_name, raw_text):

    query_clean = clean_query(query)
    package_clean = normalize_text(package_name)
    raw_clean = normalize_text(raw_text)

    if not query_clean:
        return 0.0

    # Exact phrase in package name.
    if query_clean in package_clean:
        return 1.0

    # Exact phrase in raw record.
    if query_clean in raw_clean:
        return 0.95

    query_tokens = set(query_clean.split())
    package_tokens = set(package_clean.split())
    raw_tokens = set(raw_clean.split())

    if not query_tokens:
        return 0.0

    package_overlap = (
        len(query_tokens & package_tokens)
        / len(query_tokens)
    )

    raw_overlap = (
        len(query_tokens & raw_tokens)
        / len(query_tokens)
    )

    return min(
        1.0,
        max(
            package_overlap,
            raw_overlap * 0.90,
        ),
    )


# ============================================================
# PACKAGE NAME SCORE
# ============================================================

def package_name_score(query, package_name, raw_text):

    query_clean = clean_query(query)
    package_clean = normalize_text(package_name)
    raw_clean = normalize_text(raw_text)

    query_tokens = set(query_clean.split())

    if not query_tokens:
        return 0.0

    searchable_text = package_clean + " " + raw_clean
    record_tokens = set(searchable_text.split())

    overlap = (
        len(query_tokens & record_tokens)
        / len(query_tokens)
    )

    return min(1.0, overlap)


# ============================================================
# HYBRID SCORE
# ============================================================

def calculate_hybrid_score(
    semantic,
    lexical,
    package_score,
    treatment_score,
    variant_score,
):

    # Hard treatment identity mismatch.
    if treatment_score < 0:
        return -1.0

    # Existing weighted hybrid scoring formula.
    score = (
        0.45 * semantic
        + 0.20 * lexical
        + 0.10 * package_score
        + 0.20 * treatment_score
        + 0.05 * variant_score
    )

    return score


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(query, top_k=TOP_K):

    if not isinstance(query, str) or not query.strip():
        return []

    if index.ntotal == 0:
        return []

    # --------------------------------------------------------
    # 1. Embed query using Ollama
    # --------------------------------------------------------

    query_vector = embed_query(query.strip())

    # --------------------------------------------------------
    # 2. Initial FAISS retrieval
    # --------------------------------------------------------

    candidate_k = min(
        INITIAL_RETRIEVAL_K,
        index.ntotal,
    )

    scores, indices = index.search(
        query_vector,
        candidate_k,
    )

    candidates = []

    # --------------------------------------------------------
    # 3. Calculate hybrid scores
    # --------------------------------------------------------

    for semantic_score, idx in zip(
        scores[0],
        indices[0],
    ):

        if idx == -1:
            continue

        meta = metadata[idx]

        package_name = meta.get("package_name", "")
        raw_text = meta.get("text", "")

        # Treatment identity.
        treatment_score = treatment_identity_score(
            query,
            raw_text,
        )

        # Treatment variant.
        variant = variant_score(
            query,
            raw_text,
        )

        # Lexical relevance.
        lexical = lexical_score(
            query,
            package_name,
            raw_text,
        )

        # Package relevance.
        package_score = package_name_score(
            query,
            package_name,
            raw_text,
        )

        # Weighted hybrid relevance.
        hybrid = calculate_hybrid_score(
            float(semantic_score),
            lexical,
            package_score,
            treatment_score,
            variant,
        )

        candidates.append({

            "chunk_id": int(idx),

            "hybrid_score": float(hybrid),

            "semantic_score": float(semantic_score),

            "lexical_score": float(lexical),

            "package_score": float(package_score),

            "treatment_score": float(treatment_score),

            "variant_score": float(variant),

            "package_id": meta.get("package_id", ""),

            "package_name": meta.get("package_name", ""),

            "domain": meta.get("domain", ""),

            "package_number": meta.get("package_number", ""),

            "source_pages": meta.get("source_pages", ""),

            "text": raw_text,
        })

    # --------------------------------------------------------
    # 4. Remove hard mismatches
    # --------------------------------------------------------

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["hybrid_score"] >= 0
    ]

    # --------------------------------------------------------
    # 5. Sort by hybrid score
    # --------------------------------------------------------

    valid_candidates.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return valid_candidates[:max(0, top_k)]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(query, results):

    print("\n" + "=" * 100)
    print("HYBRID PM-JAY RETRIEVAL RESULTS")
    print("=" * 100)

    print(f"\nQUERY:\n{query}")

    # Show query understanding.
    print("\nQUERY UNDERSTANDING")

    print(
        "Treatment Identity:",
        detect_treatment_identity(query),
    )

    print(
        "Treatment Variant :",
        detect_variants(query),
    )

    if not results:
        print("\nNo matching packages found.")
        return

    for i, result in enumerate(results, start=1):

        print("\n" + "-" * 100)
        print(f"RESULT {i}")
        print("-" * 100)

        print(
            f"Hybrid Score       : "
            f"{result['hybrid_score']:.4f}"
        )

        print(
            f"Semantic Score     : "
            f"{result['semantic_score']:.4f}"
        )

        print(
            f"Lexical Score      : "
            f"{result['lexical_score']:.4f}"
        )

        print(
            f"Package Score      : "
            f"{result['package_score']:.4f}"
        )

        print(
            f"Treatment Score    : "
            f"{result['treatment_score']:.4f}"
        )

        print(
            f"Variant Score      : "
            f"{result['variant_score']:.4f}"
        )

        print(f"Package ID         : {result['package_id']}")
        print(f"Package Name       : {result['package_name']}")
        print(f"Domain             : {result['domain']}")
        print(f"Package Number     : {result['package_number']}")
        print(f"Source Page        : {result['source_pages']}")

        print("\nPACKAGE RECORD:\n")
        print(result["text"][:5000])

    print("\n" + "=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("PM-JAY HYBRID RAG RETRIEVER")
    print("========================================")

    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("\nType 'exit' to stop.")

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            print("\nExiting...")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            results = retrieve(query)
            print_results(query, results)

        except Exception as error:
            print("\nRetrieval failed:")
            print(error)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
