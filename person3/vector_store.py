"""
Vector Database Module for Multimodal RAG FAQ System.
Person 3: Embeddings & Vector Database

Wraps ChromaDB -- a local, embedded, persistent vector database -- to
store and query text-chunk and image embeddings in the same semantic
space, with rich metadata for filtering (category, type, difficulty,
source_url, etc).

Why ChromaDB (vs. Pinecone / Weaviate / Qdrant Cloud)?
- No API key or hosted account needed: it runs in-process and
  persists to a local directory, so the "live vector database"
  deliverable is instantly reproducible by any teammate or grader
  without provisioning cloud infrastructure or sharing secrets.
- It speaks the same add()/query() style interface as the managed
  alternatives, so swapping to Pinecone/Qdrant/Weaviate later only
  requires changing this one module -- `indexer.py` and
  `test_retrieval.py` are unaffected.
- Cosine similarity + metadata filtering (`where=...`) out of the box,
  which is exactly what the retrieval step needs.

Connection details for this deliverable:
- "Host": local filesystem path (see `persist_directory` /
  PERSIST_DIR in README_person3.md) -- no network host/port.
- "API key": none required.
- Collections ("index names"):
    * faq_text_chunks  -- one vector per Person 2 text chunk
    * faq_images       -- one vector per unique Person 1 image
"""

import os
from typing import Any, Dict, List, Optional

import chromadb


TEXT_COLLECTION_NAME = "faq_text_chunks"
IMAGE_COLLECTION_NAME = "faq_images"


class VectorStore:
    """Thin, swappable wrapper around a persistent ChromaDB instance."""

    def __init__(self, persist_directory: str):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.text_collection = self.client.get_or_create_collection(
            name=TEXT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.image_collection = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------------------------------------------------------
    # Text chunks
    # ---------------------------------------------------------------
    def add_text_chunks(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.text_collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query_text(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.text_collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )

    # ---------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------
    def add_images(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.image_collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query_images(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.image_collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )

    # ---------------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        return {
            "text_chunks": self.text_collection.count(),
            "images": self.image_collection.count(),
        }
