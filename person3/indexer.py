"""
Indexing Pipeline for Multimodal RAG FAQ System.
Person 3: Embeddings & Vector Database

Workflow:

    Person 2 output/chunks.jsonl (text)     Person 1 input/*.json (images, base64)
                 \\                                        /
                  v                                      v
           Embed text chunks                    Embed unique images
          (CLIP text tower, batched)            (CLIP image tower)
                       \\                              /
                        v                            v
                  Upsert into ChromaDB collections
                (faq_text_chunks, faq_images) with metadata
                                |
                                v
              Write embedding_metadata.jsonl (spec format)
              Write embedding_metadata_summary.csv (human-readable)
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from .embeddings import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
    embed_texts,
    embed_images,
    is_using_fallback,
)
from .vector_store import VectorStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    """Loads Person 2's chunks.jsonl output."""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_image_lookup(input_dir: str) -> Dict[str, str]:
    """
    Scans Person 1's input JSON files and builds a mapping of
    image filename -> base64 payload. Only unique images are kept
    (a filename seen in more than one document JSON is embedded once).
    """
    lookup: Dict[str, str] = {}
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(input_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        for img in doc.get("images", []):
            filename = img.get("filename")
            b64 = img.get("base64")
            if filename and b64 and filename not in lookup:
                lookup[filename] = b64
    return lookup


def run_indexing(
    chunks_jsonl_path: str,
    input_dir: str,
    persist_directory: str,
    output_dir: str,
    text_batch_size: int = 32,
) -> Dict[str, Any]:
    print("========================================")
    print("PERSON 3: EMBEDDING & VECTOR DB INDEXING")
    print("========================================")
    print(f"Chunks JSONL:     {os.path.abspath(chunks_jsonl_path)}")
    print(f"Image input dir:  {os.path.abspath(input_dir)}")
    print(f"Vector DB path:   {os.path.abspath(persist_directory)}")
    print(f"Embedding model:  {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIMENSION}-dim)")

    chunks = load_chunks(chunks_jsonl_path)
    print(f"\nLoaded {len(chunks)} text chunks.")

    image_lookup = load_image_lookup(input_dir)
    print(f"Discovered {len(image_lookup)} unique source images.")

    store = VectorStore(persist_directory=persist_directory)

    if is_using_fallback():
        print(
            "\n[!] NOTE: real CLIP weights could not be downloaded in this "
            "environment (no network access to the model hub). Using a "
            "deterministic OFFLINE placeholder embedder so the full "
            "indexing/query pipeline can still be exercised end-to-end. "
            "Re-run with network access to produce real semantic embeddings.\n"
        )

    os.makedirs(output_dir, exist_ok=True)
    embedding_records: List[Dict[str, Any]] = []
    indexed_at = _now_iso()

    # -----------------------------------------------------------
    # 1. Embed & index text chunks
    # -----------------------------------------------------------
    text_ids, text_docs, text_metas = [], [], []
    for ch in chunks:
        meta = ch["metadata"]
        text_ids.append(ch["chunk_id"])
        text_docs.append(ch["text"])
        text_metas.append({
            "chunk_id": ch["chunk_id"],
            "parent_doc_id": ch["parent_doc_id"],
            "chunk_index": ch["chunk_index"],
            "category": meta["category"],
            "type": meta["type"],
            "difficulty": meta["difficulty"],
            "language": meta["language"],
            "has_image": meta["has_image"],
            "source_url": meta["source_url"] or "",
        })

    print(f"Embedding {len(text_docs)} text chunks...")
    all_text_embeddings: List[List[float]] = []
    for i in range(0, len(text_docs), text_batch_size):
        batch = text_docs[i:i + text_batch_size]
        all_text_embeddings.extend(embed_texts(batch))

    store.add_text_chunks(text_ids, all_text_embeddings, text_docs, text_metas)

    for cid, emb in zip(text_ids, all_text_embeddings):
        embedding_records.append({
            "chunk_id": cid,
            "embedding": emb,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding_type": "text",
            "indexed_at": indexed_at,
        })

    print(f"[OK] Indexed {len(text_ids)} text chunks into '{store.text_collection.name}'.")

    # -----------------------------------------------------------
    # 2. Embed & index images
    # -----------------------------------------------------------
    # filename -> list of chunk_ids that reference it (for parent_chunk linkage)
    image_to_chunks: Dict[str, List[str]] = {}
    for ch in chunks:
        for ref in ch["metadata"].get("image_refs", []):
            image_to_chunks.setdefault(ref, []).append(ch["chunk_id"])

    image_filenames = list(image_lookup.keys())
    image_b64_list = [image_lookup[f] for f in image_filenames]

    print(f"Embedding {len(image_filenames)} unique images...")
    image_embeddings = embed_images(image_b64_list)

    image_metas = []
    for fname in image_filenames:
        parent_chunks = image_to_chunks.get(fname, [])
        image_metas.append({
            "image_id": fname,
            "parent_chunks": ",".join(parent_chunks),
        })

    store.add_images(image_filenames, image_embeddings, image_metas)

    for fname, emb in zip(image_filenames, image_embeddings):
        parent_chunks = image_to_chunks.get(fname, [])
        embedding_records.append({
            "image_id": fname,
            "embedding": emb,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding_type": "image",
            "parent_chunk": parent_chunks[0] if parent_chunks else None,
            "indexed_at": indexed_at,
        })

    print(f"[OK] Indexed {len(image_filenames)} images into '{store.image_collection.name}'.")

    # -----------------------------------------------------------
    # 3. Write deliverable metadata files
    # -----------------------------------------------------------
    meta_jsonl_path = os.path.join(output_dir, "embedding_metadata.jsonl")
    with open(meta_jsonl_path, "w", encoding="utf-8") as f:
        for rec in embedding_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[OK] Wrote embedding metadata JSONL: {meta_jsonl_path}")

    csv_path = os.path.join(output_dir, "embedding_metadata_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "embedding_type", "embedding_model", "embedding_dimension",
            "parent_chunk", "indexed_at",
        ])
        for rec in embedding_records:
            rid = rec.get("chunk_id") or rec.get("image_id")
            writer.writerow([
                rid,
                rec["embedding_type"],
                rec["embedding_model"],
                rec["embedding_dimension"],
                rec.get("parent_chunk", ""),
                rec["indexed_at"],
            ])
    print(f"[OK] Wrote embedding metadata summary CSV: {csv_path}")

    counts = store.counts()
    print("\n========================================")
    print("INDEXING COMPLETE")
    print("========================================")
    print(f"Text chunks indexed: {counts['text_chunks']}")
    print(f"Images indexed:      {counts['images']}")

    return {
        "text_chunks_indexed": counts["text_chunks"],
        "images_indexed": counts["images"],
        "persist_directory": os.path.abspath(persist_directory),
        "embedding_metadata_jsonl": meta_jsonl_path,
        "embedding_metadata_csv": csv_path,
        "using_fallback_embedder": is_using_fallback(),
    }
