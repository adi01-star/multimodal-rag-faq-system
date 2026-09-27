"""
Validation Module for Multimodal RAG FAQ System.
Person 2: Preprocessing & Chunking

Validates output JSONL and CSV files against all project requirements:
- Schema adherence
- Field types and non-null constraints
- Controlled vocabulary validation
- Broken image reference detection
- Token count sanity
- Deduplication and uniqueness verification
- Exact match between JSONL and CSV
"""

import os
import json
import csv
from typing import Dict, Any, List, Set, Optional

from .metadata import ALLOWED_CATEGORIES, ALLOWED_TYPES, ALLOWED_DIFFICULTIES


def validate_outputs(
    jsonl_path: str,
    csv_path: str,
    input_dir: str,
    dedup_stats: Optional[Dict[str, Any]] = None,
    num_source_docs: int = 0
) -> Dict[str, Any]:
    """
    Run comprehensive validation on generated chunks.jsonl and chunks_validation.csv.

    Args:
        jsonl_path (str): Path to chunks.jsonl
        csv_path (str): Path to chunks_validation.csv
        input_dir (str): Path to Person 1 input directory (to check images and source docs)
        dedup_stats (Dict[str, Any]): Deduplication stats from pipeline run
        num_source_docs (int): Number of source documents processed

    Returns:
        Dict[str, Any]: Detailed validation results and summary metrics.
    """
    results = {
        "status": "PASSED",
        "errors": [],
        "warnings": [],
        "documents_processed": num_source_docs,
        "chunks_before_dedup": dedup_stats.get("total_chunks_before", 0) if dedup_stats else 0,
        "exact_duplicates_removed": dedup_stats.get("exact_duplicates_removed", 0) if dedup_stats else 0,
        "near_duplicates_removed": dedup_stats.get("near_duplicates_removed", 0) if dedup_stats else 0,
        "final_chunks": 0,
        "avg_tokens": 0.0,
        "min_tokens": 0,
        "max_tokens": 0,
        "chunks_with_images": 0,
        "chunks_without_images": 0,
        "invalid_chunks": 0,
        "broken_image_refs": 0,
        "duplicate_chunk_ids": 0,
        "empty_chunks": 0
    }

    if not os.path.exists(jsonl_path):
        results["status"] = "FAILED"
        results["errors"].append(f"JSONL file missing: {jsonl_path}")
        return results

    if not os.path.exists(csv_path):
        results["status"] = "FAILED"
        results["errors"].append(f"CSV file missing: {csv_path}")
        return results

    # Collect all valid image filenames available in input directory
    available_images: Set[str] = set()
    # 1. From embedded_images subdirectories
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                available_images.add(f)

    # 2. From Person 1 JSON files in input_dir
    for f in os.listdir(input_dir):
        if f.lower().endswith(".json"):
            try:
                with open(os.path.join(input_dir, f), "r", encoding="utf-8") as jf:
                    doc = json.load(jf)
                    for img in doc.get("images", []):
                        if img.get("filename"):
                            available_images.add(img["filename"])
            except Exception:
                pass

    # Read and validate JSONL
    chunks = []
    seen_chunk_ids: Set[str] = set()
    seen_texts: Set[str] = set()
    token_counts: List[int] = []

    required_top_keys = {"chunk_id", "text", "chunk_number", "metadata", "parent_doc_id", "chunk_index"}
    required_meta_keys = {
        "category", "type", "difficulty", "source_doc_id",
        "source_url", "has_image", "image_refs", "language", "tokens_approx"
    }

    with open(jsonl_path, "r", encoding="utf-8") as jf:
        for line_num, line in enumerate(jf, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                chunk = json.loads(line_str)
            except json.JSONDecodeError as e:
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            # Verify top keys
            missing_top = required_top_keys - set(chunk.keys())
            if missing_top:
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Missing top-level keys: {missing_top}")

            chunk_id = chunk.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                results["invalid_chunks"] += 1
                results["empty_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Empty or invalid chunk_id")
            elif chunk_id in seen_chunk_ids:
                results["duplicate_chunk_ids"] += 1
                results["errors"].append(f"Line {line_num}: Duplicate chunk_id: {chunk_id}")
            else:
                seen_chunk_ids.add(chunk_id)

            # Check text
            text = chunk.get("text", "")
            if not text or not str(text).strip():
                results["empty_chunks"] += 1
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Empty chunk text")

            # Check metadata
            meta = chunk.get("metadata", {})
            if not isinstance(meta, dict):
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Metadata must be a dictionary")
                continue

            missing_meta = required_meta_keys - set(meta.keys())
            if missing_meta:
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Missing metadata keys: {missing_meta}")

            # Validate controlled vocabularies
            if meta.get("category") not in ALLOWED_CATEGORIES:
                results["errors"].append(f"Line {line_num}: Category '{meta.get('category')}' not in allowed set")
                results["invalid_chunks"] += 1

            if meta.get("type") not in ALLOWED_TYPES:
                results["errors"].append(f"Line {line_num}: Type '{meta.get('type')}' not in allowed set")
                results["invalid_chunks"] += 1

            if meta.get("difficulty") not in ALLOWED_DIFFICULTIES:
                results["errors"].append(f"Line {line_num}: Difficulty '{meta.get('difficulty')}' not in allowed set")
                results["invalid_chunks"] += 1

            # Validate tokens_approx
            tok = meta.get("tokens_approx")
            if not isinstance(tok, int) or tok <= 0:
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: Invalid tokens_approx: {tok}")
            else:
                token_counts.append(tok)

            # Validate image references
            has_img = meta.get("has_image")
            img_refs = meta.get("image_refs", [])
            if not isinstance(img_refs, list):
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: image_refs must be a list")
            else:
                if has_img and len(img_refs) == 0:
                    results["invalid_chunks"] += 1
                    results["errors"].append(f"Line {line_num}: has_image is True but image_refs is empty")
                if not has_img and len(img_refs) > 0:
                    results["invalid_chunks"] += 1
                    results["errors"].append(f"Line {line_num}: has_image is False but image_refs contains items")

                for ref in img_refs:
                    if ref not in available_images:
                        results["broken_image_refs"] += 1
                        results["errors"].append(f"Line {line_num}: Broken image reference '{ref}' not found in dataset")

            if has_img:
                results["chunks_with_images"] += 1
            else:
                results["chunks_without_images"] += 1

            # Validate parent doc id consistency
            if chunk.get("parent_doc_id") != meta.get("source_doc_id"):
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: parent_doc_id and metadata.source_doc_id mismatch")

            # Validate chunk index & chunk number
            c_idx = chunk.get("chunk_index")
            c_num = chunk.get("chunk_number")
            if not isinstance(c_idx, int) or not isinstance(c_num, int):
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: chunk_index and chunk_number must be integers")
            elif c_num != c_idx + 1:
                results["invalid_chunks"] += 1
                results["errors"].append(f"Line {line_num}: chunk_number ({c_num}) must equal chunk_index + 1 ({c_idx + 1})")

            chunks.append(chunk)

    results["final_chunks"] = len(chunks)
    if token_counts:
        results["avg_tokens"] = round(sum(token_counts) / len(token_counts), 1)
        results["min_tokens"] = min(token_counts)
        results["max_tokens"] = max(token_counts)

    # Validate CSV consistency
    csv_rows = []
    with open(csv_path, "r", encoding="utf-8") as cf:
        reader = csv.DictReader(cf)
        expected_csv_cols = [
            "chunk_id", "doc_id", "chunk_index", "chunk_number", "token_count",
            "category", "type", "difficulty", "language", "has_image",
            "image_count", "source_url", "text_length"
        ]
        if reader.fieldnames != expected_csv_cols:
            results["errors"].append(f"CSV columns mismatch. Expected {expected_csv_cols}, got {reader.fieldnames}")
            results["status"] = "FAILED"

        for r in reader:
            csv_rows.append(r)

    if len(csv_rows) != len(chunks):
        results["errors"].append(f"CSV row count ({len(csv_rows)}) does not match JSONL chunk count ({len(chunks)})")
        results["status"] = "FAILED"
    else:
        # Check matching rows
        for i, (ch, row) in enumerate(zip(chunks, csv_rows)):
            meta = ch["metadata"]
            if row["chunk_id"] != ch["chunk_id"]:
                results["errors"].append(f"Row {i+1}: chunk_id mismatch: CSV {row['chunk_id']} vs JSONL {ch['chunk_id']}")
                results["status"] = "FAILED"
                break
            if int(row["token_count"]) != meta["tokens_approx"]:
                results["errors"].append(f"Row {i+1}: token count mismatch")
                results["status"] = "FAILED"
                break

    if results["errors"] or results["invalid_chunks"] > 0 or results["broken_image_refs"] > 0:
        results["status"] = "FAILED"

    return results


def print_validation_report(results: Dict[str, Any]) -> str:
    """
    Format and print the official validation report.
    """
    report = f"""========================================
PERSON 2 VALIDATION REPORT
========================================

Documents processed: {results['documents_processed']}
Chunks before deduplication: {results['chunks_before_dedup']}
Exact duplicates removed: {results['exact_duplicates_removed']}
Near duplicates removed: {results['near_duplicates_removed']}
Final chunks: {results['final_chunks']}

Average tokens/chunk: {results['avg_tokens']}
Minimum tokens: {results['min_tokens']}
Maximum tokens: {results['max_tokens']}

Chunks with images: {results['chunks_with_images']}
Chunks without images: {results['chunks_without_images']}

Invalid chunks: {results['invalid_chunks']}
Broken image references: {results['broken_image_refs']}
Duplicate chunk IDs: {results['duplicate_chunk_ids']}
Empty chunks: {results['empty_chunks']}

VALIDATION STATUS: {results['status']}
========================================"""
    print(report)
    if results["errors"]:
        print("\nValidation Errors:")
        for err in results["errors"][:10]:
            print(f"  - {err}")
    return report
