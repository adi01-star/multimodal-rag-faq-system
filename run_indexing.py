#!/usr/bin/env python3
"""
Indexing Runner for Multimodal RAG FAQ System.
Person 3: Embeddings & Vector Database

Usage:
    python run_indexing.py
    python run_indexing.py --chunks_jsonl output/chunks.jsonl --input_dir input --persist_dir vector_db
"""

import argparse
import os

from person3.indexer import run_indexing


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Person 3: Embeddings & Vector Database Indexing Pipeline")
    parser.add_argument("--chunks_jsonl", default=os.path.join(here, "output", "chunks.jsonl"),
                         help="Path to Person 2's chunks.jsonl")
    parser.add_argument("--input_dir", default=os.path.join(here, "input"),
                         help="Path to Person 1's input directory (contains *.json with base64 images)")
    parser.add_argument("--persist_dir", default=os.path.join(here, "vector_db"),
                         help="Directory where the local ChromaDB vector database is persisted")
    parser.add_argument("--output_dir", default=os.path.join(here, "output"),
                         help="Directory to write embedding_metadata.jsonl / .csv")
    args = parser.parse_args()

    run_indexing(
        chunks_jsonl_path=args.chunks_jsonl,
        input_dir=args.input_dir,
        persist_directory=args.persist_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
