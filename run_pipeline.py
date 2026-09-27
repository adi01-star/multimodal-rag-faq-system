#!/usr/bin/env python3
"""
Pipeline Runner for Multimodal RAG FAQ System.
Person 2: Preprocessing & Chunking

Pipeline Workflow:
Person 1 JSON files
    ↓
Load JSON
    ↓
Clean and normalize text
    ↓
Detect logical sections / FAQ structure
    ↓
Chunk text (250-500 tokens target)
    ↓
Generate metadata
    ↓
Preserve image references
    ↓
Remove duplicate / near-duplicate chunks
    ↓
Estimate token count
    ↓
Validate output
    ↓
Generate JSONL
    ↓
Generate CSV
    ↓
Generate Report
"""

import os
import sys

# Ensure UTF-8 stdout on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import csv
import argparse
from typing import List, Dict, Any

from src.preprocess import clean_text
from src.chunking import chunk_document, count_tokens
from src.metadata import build_chunk_metadata
from src.deduplication import deduplicate_chunks
from src.validation import validate_outputs, print_validation_report


def process_pipeline(
    input_dir: str,
    output_dir: str,
    similarity_threshold: float = 0.88,
    target_min_tokens: int = 250,
    target_max_tokens: int = 500
) -> Dict[str, Any]:
    """
    Executes the complete Person 2 preprocessing and chunking pipeline.
    """
    print("========================================")
    print("STARTING PERSON 2 PREPROCESSING PIPELINE")
    print("========================================")
    print(f"Input Directory:  {os.path.abspath(input_dir)}")
    print(f"Output Directory: {os.path.abspath(output_dir)}")
    print(f"Target Chunk Range: {target_min_tokens} - {target_max_tokens} tokens")
    print(f"Deduplication Similarity Threshold: {similarity_threshold}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Discover Person 1 JSON files
    json_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".json")
    ])

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    print(f"\nDiscovered {len(json_files)} source document JSON files.")

    all_raw_chunks: List[Dict[str, Any]] = []
    global_chunk_counter = 1

    # 2. Process each document
    for doc_idx, json_path in enumerate(json_files, start=1):
        filename = os.path.basename(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_id = doc_data.get("id") or os.path.splitext(filename)[0]
        raw_text = doc_data.get("raw_text", "")
        source_url = doc_data.get("source_url")
        source_title = doc_data.get("source_title", filename)
        doc_images = doc_data.get("images", [])

        # Clean and normalize text
        cleaned_text = clean_text(raw_text)

        # Chunk text
        chunks_info = chunk_document(
            cleaned_text,
            target_min=target_min_tokens,
            target_max=target_max_tokens
        )

        # Build chunks with initial metadata and index per document
        for c_idx, ch in enumerate(chunks_info):
            c_text = ch["text"]
            sec_type = ch["section_type"]
            tokens = ch["tokens_approx"]

            metadata = build_chunk_metadata(
                chunk_text=c_text,
                chunk_index=c_idx,
                section_type=sec_type,
                doc_id=doc_id,
                source_url=source_url,
                doc_title=source_title,
                doc_images=doc_images,
                tokens_approx=tokens,
                raw_doc_data=doc_data
            )

            chunk_obj = {
                "chunk_id": f"chunk_{global_chunk_counter:03d}_v1",
                "text": c_text,
                "chunk_number": c_idx + 1,
                "metadata": metadata,
                "parent_doc_id": doc_id,
                "chunk_index": c_idx
            }
            all_raw_chunks.append(chunk_obj)
            global_chunk_counter += 1

    print(f"Generated {len(all_raw_chunks)} candidate chunks before deduplication.")

    # 3. Deduplicate chunks
    print("\nRunning exact and near-duplicate detection...")
    unique_chunks, dedup_stats = deduplicate_chunks(
        all_raw_chunks,
        near_dup_threshold=similarity_threshold
    )

    print(f"Exact duplicates removed: {dedup_stats['exact_duplicates_removed']}")
    print(f"Near duplicates removed:  {dedup_stats['near_duplicates_removed']}")
    print(f"Remaining final chunks:   {len(unique_chunks)}")

    # 4. Re-index per parent document to ensure chunk_index (0-indexed)
    # and chunk_number (1-indexed) are strictly contiguous per parent document
    doc_chunk_counters: Dict[str, int] = {}
    for ch in unique_chunks:
        pid = ch["parent_doc_id"]
        cur_idx = doc_chunk_counters.get(pid, 0)
        ch["chunk_index"] = cur_idx
        ch["chunk_number"] = cur_idx + 1
        doc_chunk_counters[pid] = cur_idx + 1

    # Re-assign globally unique chunk_id to keep clean sequential numbering
    for i, ch in enumerate(unique_chunks, start=1):
        ch["chunk_id"] = f"chunk_{i:03d}_v1"

    # 5. Write output JSONL
    jsonl_path = os.path.join(output_dir, "chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for ch in unique_chunks:
            jf.write(json.dumps(ch, ensure_ascii=False) + "\n")
    print(f"\n[OK] Generated JSONL: {jsonl_path}")

    # 6. Write output CSV
    csv_path = os.path.join(output_dir, "chunks_validation.csv")
    csv_fieldnames = [
        "chunk_id",
        "doc_id",
        "chunk_index",
        "chunk_number",
        "token_count",
        "category",
        "type",
        "difficulty",
        "language",
        "has_image",
        "image_count",
        "source_url",
        "text_length"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
        writer.writeheader()
        for ch in unique_chunks:
            meta = ch["metadata"]
            writer.writerow({
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["parent_doc_id"],
                "chunk_index": ch["chunk_index"],
                "chunk_number": ch["chunk_number"],
                "token_count": meta["tokens_approx"],
                "category": meta["category"],
                "type": meta["type"],
                "difficulty": meta["difficulty"],
                "language": meta["language"],
                "has_image": meta["has_image"],
                "image_count": len(meta.get("image_refs", [])),
                "source_url": meta["source_url"] if meta["source_url"] is not None else "",
                "text_length": len(ch["text"])
            })
    print(f"[OK] Generated CSV:   {csv_path}")

    # 7. Run validation
    print("\nRunning validation suite...")
    val_results = validate_outputs(
        jsonl_path=jsonl_path,
        csv_path=csv_path,
        input_dir=input_dir,
        dedup_stats=dedup_stats,
        num_source_docs=len(json_files)
    )

    print("\n")
    print_validation_report(val_results)

    if val_results["status"] != "PASSED":
        print("\nPipeline failed validation!", file=sys.stderr)
        sys.exit(1)

    return val_results


def main():
    parser = argparse.ArgumentParser(description="Person 2: Preprocessing and Chunking Pipeline")
    parser.add_argument(
        "--input_dir",
        default=os.path.join(os.path.dirname(__file__), "input"),
        help="Path to directory containing Person 1 JSON files"
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
        help="Path to output directory"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.88,
        help="Cosine similarity threshold for near-duplicate removal"
    )
    parser.add_argument(
        "--target_min",
        type=int,
        default=250,
        help="Target minimum tokens per chunk"
    )
    parser.add_argument(
        "--target_max",
        type=int,
        default=500,
        help="Target maximum tokens per chunk"
    )

    args = parser.parse_args()

    process_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.similarity_threshold,
        target_min_tokens=args.target_min,
        target_max_tokens=args.target_max
    )


if __name__ == "__main__":
    main()
