"""
Deduplication Module for Multimodal RAG FAQ System.
Person 2: Preprocessing & Chunking

Implements:
1. Exact duplicate detection using SHA-256 hash of normalized text.
2. Near-duplicate detection using TF-IDF and Cosine Similarity (threshold: 0.88).
Embeddings are intentionally NOT used as they are assigned to Person 3.
"""

import hashlib
import re
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_for_hashing(text: str) -> str:
    """
    Aggressively normalize text for exact duplicate detection.
    Lowercases, removes all punctuation, and collapses whitespace.
    """
    lowered = text.lower()
    # Strip non-alphanumeric characters
    cleaned = re.sub(r"[^\w\s]", "", lowered)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", cleaned).strip()
    return normalized


def compute_sha256(text: str) -> str:
    """
    Compute SHA-256 hash of normalized text.
    """
    norm = normalize_for_hashing(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    near_dup_threshold: float = 0.88
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Removes exact and near-duplicate chunks.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk objects.
        near_dup_threshold (float): Cosine similarity threshold for near-duplicate removal (default 0.88).

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            - deduplicated_chunks: Filtered list of unique chunks.
            - stats: Dictionary tracking deduplication metrics.
    """
    total_before = len(chunks)
    exact_duplicates_removed = 0
    near_duplicates_removed = 0
    exact_removed_info = []
    near_removed_info = []

    # -------------------------------------------------------------
    # Pass 1: Exact Duplicate Detection (SHA-256)
    # -------------------------------------------------------------
    seen_hashes: Dict[str, Dict[str, Any]] = {}
    pass1_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        ch_hash = compute_sha256(chunk["text"])
        if ch_hash in seen_hashes:
            exact_duplicates_removed += 1
            original = seen_hashes[ch_hash]
            exact_removed_info.append({
                "removed_chunk_id": chunk.get("chunk_id"),
                "kept_chunk_id": original.get("chunk_id"),
                "snippet": chunk["text"][:80]
            })
        else:
            seen_hashes[ch_hash] = chunk
            pass1_chunks.append(chunk)

    # -------------------------------------------------------------
    # Pass 2: Near-Duplicate Detection (TF-IDF + Cosine Similarity)
    # -------------------------------------------------------------
    if not pass1_chunks or len(pass1_chunks) == 1:
        final_chunks = pass1_chunks
    else:
        final_chunks = []
        final_texts = []

        for chunk in pass1_chunks:
            chunk_text = chunk["text"]
            if not final_texts:
                final_chunks.append(chunk)
                final_texts.append(chunk_text)
                continue

            # Compare current chunk against already accepted chunks
            corpus = final_texts + [chunk_text]
            try:
                vectorizer = TfidfVectorizer(ngram_range=(1, 1))
                tfidf_matrix = vectorizer.fit_transform(corpus)
                # Compute cosine similarity of the last item with all previous items
                current_vec = tfidf_matrix[-1]
                prev_vecs = tfidf_matrix[:-1]
                similarities = cosine_similarity(current_vec, prev_vecs)[0]

                max_sim = float(max(similarities)) if len(similarities) > 0 else 0.0
                best_match_idx = int(similarities.argmax()) if len(similarities) > 0 else -1

                if max_sim >= near_dup_threshold:
                    near_duplicates_removed += 1
                    near_removed_info.append({
                        "removed_chunk_id": chunk.get("chunk_id"),
                        "matched_chunk_id": final_chunks[best_match_idx].get("chunk_id"),
                        "similarity": round(max_sim, 4),
                        "snippet": chunk_text[:80]
                    })
                else:
                    final_chunks.append(chunk)
                    final_texts.append(chunk_text)
            except Exception:
                # Fallback if vocabulary is empty
                final_chunks.append(chunk)
                final_texts.append(chunk_text)

    stats = {
        "total_chunks_before": total_before,
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicates_removed": near_duplicates_removed,
        "final_chunks": len(final_chunks),
        "exact_removed_details": exact_removed_info,
        "near_removed_details": near_removed_info,
        "near_dup_threshold": near_dup_threshold
    }

    return final_chunks, stats
