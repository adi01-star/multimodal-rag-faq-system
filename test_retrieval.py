#!/usr/bin/env python3
"""
Sample retrieval test queries for Person 3's vector database.
Person 3: Embeddings & Vector Database

Runs a handful of representative FAQ questions against the indexed
text-chunk collection and reports whether the expected source document
appears in the top-K results -- a simple proxy for the project's
"Top-3 retrieval accuracy >85%" success criterion.

NOTE: If real CLIP weights aren't available in this environment
(no network access to the model hub), the offline fallback embedder
makes retrieval quality meaningless (vectors are pseudo-random). In
that case this script still verifies the pipeline runs end-to-end;
accuracy numbers only become meaningful once re-run with real network
access to CLIP.

Usage:
    python test_retrieval.py
    python test_retrieval.py --persist_dir vector_db --top_k 3
"""

import argparse
import os
import sys

from person3.embeddings import embed_text, is_using_fallback
from person3.vector_store import VectorStore


SAMPLE_QUERIES = [
    {
        "query": "How do I reset my forgotten password?",
        "expected_doc_id": "01_account_and_password_faq",
    },
    {
        "query": "My internet keeps disconnecting, what should I do?",
        "expected_doc_id": "02_network_troubleshooting_faq",
    },
    {
        "query": "Where can I safely download software before installing it?",
        "expected_doc_id": "03_software_installation_faq",
    },
    {
        "query": "How can I tell if a login page is a phishing scam?",
        "expected_doc_id": "04_security_and_safe_usage_faq",
    },
    {
        "query": "What is Retrieval-Augmented Generation and why is it multimodal?",
        "expected_doc_id": "05_multimodal_rag_faq_system_faq",
    },
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Person 3: Sample retrieval smoke tests")
    parser.add_argument("--persist_dir", default=os.path.join(here, "vector_db"))
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    store = VectorStore(persist_directory=args.persist_dir)
    counts = store.counts()
    if counts["text_chunks"] == 0:
        print("No text chunks indexed yet. Run run_indexing.py first.", file=sys.stderr)
        sys.exit(1)

    if is_using_fallback():
        print(
            "[!] WARNING: running with the OFFLINE FALLBACK embedder -- "
            "results below are NOT semantically meaningful, this is a "
            "pipeline smoke test only.\n"
        )

    hits = 0
    for case in SAMPLE_QUERIES:
        q_emb = embed_text(case["query"])
        results = store.query_text(q_emb, top_k=args.top_k)

        result_chunk_ids = results["ids"][0]
        result_doc_ids = [m["parent_doc_id"] for m in results["metadatas"][0]]
        hit = case["expected_doc_id"] in result_doc_ids
        hits += int(hit)

        print(f"Query: {case['query']!r}")
        print(f"  Expected source doc: {case['expected_doc_id']}")
        print(f"  Top-{args.top_k} chunk_ids:  {result_chunk_ids}")
        print(f"  Top-{args.top_k} source docs: {result_doc_ids}")
        print(f"  {'HIT' if hit else 'MISS'}")
        print()

    accuracy = hits / len(SAMPLE_QUERIES) * 100
    print(f"Top-{args.top_k} retrieval accuracy on sample queries: {accuracy:.1f}% ({hits}/{len(SAMPLE_QUERIES)})")


if __name__ == "__main__":
    main()
