
import os
import re
import pickle

import faiss
import numpy as np
import ollama

from tqdm import tqdm

from config import (
    EMBEDDING_MODEL,
    DATA_PATH,
    INDEX_PATH,
    METADATA_PATH,
)


# ==================================================
# LOCAL OLLAMA E5 EMBEDDINGS
# ==================================================

def embed_one_text(text, prefix="passage: "):
    """Create one embedding using the local Ollama E5 model."""
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=prefix + text,
    )

    # Current Ollama Python API returns an embeddings array.
    if hasattr(response, "embeddings"):
        raw_vector = response.embeddings[0]
    else:
        raw_vector = response["embeddings"][0]

    vector = np.asarray(raw_vector, dtype="float32").reshape(-1)

    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Ollama returned an empty or invalid embedding.")

    return vector


# ==================================================
# LOAD DOCUMENT
# ==================================================

def load_document():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found:\n{DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    print(f"Loaded document: {len(text):,} characters")
    return text


# ==================================================
# EXTRACT PACKAGE RECORDS
# ==================================================

def extract_package_records(text):
    """Find opening PACKAGE RECORD markers at the start of a line."""
    pattern = re.compile(r"(?m)^PACKAGE RECORD\s*$")
    matches = list(pattern.finditer(text))

    records = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        record = text[start:end].strip()

        record = re.sub(
            r"\nEND PACKAGE RECORD\s*$",
            "",
            record,
            flags=re.MULTILINE,
        ).strip()

        if len(record) >= 100:
            records.append(record)

    return records


# ==================================================
# EXTRACT FIELD
# ==================================================

def extract_field(text, field_name):
    """Extract FIELD: value from a package record."""
    pattern = re.compile(
        rf"(?im)^{re.escape(field_name)}\s*:\s*(.+)$"
    )
    match = pattern.search(text)

    return match.group(1).strip() if match else ""


# ==================================================
# EXTRACT PACKAGE NAME
# ==================================================

def extract_package_name(record):
    source = extract_field(record, "PACKAGE SOURCE DETAILS")

    if not source:
        marker = "PACKAGE SOURCE DETAILS:"
        if marker in record:
            source = record.split(marker, 1)[1].strip()

    if not source:
        return "Unknown"

    amount_pattern = re.compile(
        r"(?:Rs\.?\s*)?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b"
    )
    match = amount_pattern.search(source)

    if match:
        name = source[:match.start()].strip()
        name = re.sub(r"\s+", " ", name).strip(" -:|")

        if len(name) >= 3:
            return name

    first_line = source.splitlines()[0].strip()
    return first_line if first_line else "Unknown"


# ==================================================
# EXTRACT VARIANT TERMS
# ==================================================

def extract_variant_terms(package_name):
    known_variants = [
        "palliative",
        "radical",
        "adjuvant",
        "neoadjuvant",
        "imrt",
        "igrt",
        "srt",
        "srs",
        "3d crt",
        "2d planning",
        "external beam",
        "brachytherapy",
    ]

    name_lower = package_name.lower()
    return [variant for variant in known_variants if variant in name_lower]


# ==================================================
# EXTRACT METADATA
# ==================================================

def extract_metadata(record, chunk_id):
    package_id = extract_field(record, "PACKAGE ID")
    domain = extract_field(record, "DOMAIN")
    table_section = extract_field(record, "TABLE SECTION")
    package_number = extract_field(record, "PACKAGE NUMBER")
    source_pages = extract_field(record, "SOURCE PAGE(S)")
    empanelment = extract_field(record, "EMPANELMENT CLASSIFICATION")
    pre_auth = extract_field(record, "PRE-AUTHORIZATION POLICY")
    package_name = extract_package_name(record)
    variants = extract_variant_terms(package_name)

    return {
        "chunk_id": chunk_id,
        "package_id": package_id,
        "domain": domain,
        "table_section": table_section,
        "package_number": package_number,
        "package_name": package_name,
        "variant_terms": variants,
        "source_pages": source_pages,
        "empanelment_classification": empanelment,
        "pre_authorization_policy": pre_auth,
        "text": record,
    }


# ==================================================
# SPLIT LONG RECORD
# ==================================================

def split_long_record(record, max_chars=3500):
    """Keep ordinary records intact; split oversized records by line."""
    if len(record) <= max_chars:
        return [record]

    lines = record.splitlines()
    metadata_lines = []
    body_lines = []
    found_body = False

    for line in lines:
        if line.strip() == "PACKAGE SOURCE DETAILS:":
            found_body = True
            metadata_lines.append(line)
        elif not found_body:
            metadata_lines.append(line)
        else:
            body_lines.append(line)

    metadata = "\n".join(metadata_lines).strip()
    chunks = []
    current = metadata

    for line in body_lines:
        line = line.strip()
        if not line:
            continue

        candidate = current + "\n" + line

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if len(current.strip()) > 100:
                chunks.append(current.strip())

            current = metadata + "\n" + line

    if len(current.strip()) > 100:
        chunks.append(current.strip())

    return chunks


# ==================================================
# CREATE CHUNKS + METADATA
# ==================================================

def create_chunks(text):
    records = extract_package_records(text)

    print(f"Found {len(records)} PACKAGE RECORD sections.")

    chunks = []
    metadata = []
    chunk_id = 0

    for record in records:
        pieces = split_long_record(record)

        for piece in pieces:
            chunks.append(piece)
            metadata.append(extract_metadata(piece, chunk_id))
            chunk_id += 1

    return chunks, metadata


# ==================================================
# OLLAMA E5 EMBEDDINGS
# ==================================================

def embed_texts(texts):
    embeddings = []
    expected_dimension = None

    for i, text in enumerate(
        tqdm(texts, desc="Creating local Ollama E5 embeddings")
    ):
        try:
            vector = embed_one_text(text, prefix="passage: ")

            if expected_dimension is None:
                expected_dimension = vector.shape[0]
            elif vector.shape[0] != expected_dimension:
                raise ValueError(
                    f"Inconsistent embedding dimension at chunk {i}: "
                    f"expected {expected_dimension}, got {vector.shape[0]}"
                )

            embeddings.append(vector)

        except Exception as error:
            print(f"\nEmbedding failed for chunk {i}.")
            print(f"Error: {error}")
            raise

    if not embeddings:
        raise RuntimeError("No embeddings were generated.")

    return np.vstack(embeddings).astype("float32")


# ==================================================
# NORMALIZE
# ==================================================

def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embeddings / norms


# ==================================================
# BUILD FAISS
# ==================================================

def build_faiss(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(INDEX_PATH) or ".", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

    return index


# ==================================================
# SAVE METADATA
# ==================================================

def save_metadata(metadata):
    os.makedirs(os.path.dirname(METADATA_PATH) or ".", exist_ok=True)

    with open(METADATA_PATH, "wb") as file:
        pickle.dump(metadata, file)


# ==================================================
# MAIN
# ==================================================

def main():
    print("\n========================================")
    print("PM-JAY HYBRID RAG INGESTION")
    print("Local Ollama E5 + FAISS")
    print("========================================")

    print("\nLoading PM-JAY knowledge base...")
    text = load_document()

    print("\nCreating package-aware chunks...")
    chunks, metadata = create_chunks(text)

    print(f"\nCreated {len(chunks)} RAG chunks.")

    if not chunks:
        raise RuntimeError("No PACKAGE RECORD chunks found.")

    print("\n========================================")
    print("FIRST PACKAGE METADATA")
    print("========================================")

    first = metadata[0]
    print(f"Package ID   : {first['package_id']}")
    print(f"Domain       : {first['domain']}")
    print(f"Package No.  : {first['package_number']}")
    print(f"Package Name : {first['package_name']}")
    print(f"Source Page  : {first['source_pages']}")

    print("\nCreating local Ollama E5 embeddings...")
    embeddings = embed_texts(chunks)

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"Embedding count ({len(embeddings)}) does not match "
            f"metadata count ({len(metadata)})."
        )

    print("\nEmbedding shape:", embeddings.shape)

    embeddings = normalize_embeddings(embeddings)

    # Save new files only after embeddings have completed successfully.
    # Back up the old index and metadata before running this script.
    print("\nBuilding FAISS index...")
    index = build_faiss(embeddings)

    print("\nSaving metadata...")
    save_metadata(metadata)

    print("\n========================================")
    print("HYBRID RAG DATABASE CREATED")
    print("========================================")
    print(f"Total chunks : {index.ntotal}")
    print(f"Vector size  : {embeddings.shape[1]}")
    print(f"FAISS index  : {INDEX_PATH}")
    print(f"Metadata     : {METADATA_PATH}")


if __name__ == "__main__":
    main()
