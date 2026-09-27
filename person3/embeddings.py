"""
Embedding Module for Multimodal RAG FAQ System.
Person 3: Embeddings & Vector Database

Provides a unified multimodal embedding interface using CLIP
(via sentence-transformers' "clip-ViT-B-32"), which embeds both
text and images into the SAME shared 512-dimensional vector space.
This is what makes cross-modal retrieval possible (e.g. a text
query can retrieve a relevant screenshot, and vice versa).

Why CLIP / clip-ViT-B-32?
- It is the standard, well-supported multimodal embedding model
  recommended in the project brief (CLIP, Sentence-BERT, or a
  domain-fine-tuned variant).
- sentence-transformers wraps it with an identical .encode() API
  for both PIL images and text strings, which keeps this module
  simple and keeps text/image vectors comparable.
- 512-dim output keeps the index small and fast for a 2,000+ chunk
  / 500+ image corpus.

Offline fallback:
If the pretrained weights cannot be downloaded (no network access to
the model hub), this module transparently falls back to a
deterministic pseudo-embedder so the rest of the pipeline (chunking
-> embedding -> indexing -> retrieval) can still be built, tested,
and demonstrated end-to-end. The fallback is clearly logged and is
NOT semantically meaningful -- swap back to the real model by simply
running in an environment with network access; no other code changes
are required.
"""

import base64
import hashlib
import io
import sys
from typing import List

import numpy as np

EMBEDDING_MODEL_NAME = "clip-ViT-B-32"
EMBEDDING_DIMENSION = 512

_model = None
_USING_FALLBACK = False
_FALLBACK_WARNED = False


def _load_model() -> None:
    global _model, _USING_FALLBACK
    if _model is not None or _USING_FALLBACK:
        return
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[embeddings] Loaded multimodal embedding model: {EMBEDDING_MODEL_NAME}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - want to catch any load failure
        _USING_FALLBACK = True
        _warn_fallback(exc)


def _warn_fallback(exc: Exception) -> None:
    global _FALLBACK_WARNED
    if _FALLBACK_WARNED:
        return
    _FALLBACK_WARNED = True
    print(
        f"[embeddings] WARNING: could not load '{EMBEDDING_MODEL_NAME}' "
        f"({exc.__class__.__name__}: {exc}). Falling back to a deterministic "
        f"OFFLINE pseudo-embedder. This fallback exists purely so the "
        f"embedding/indexing/retrieval pipeline can be exercised without "
        f"network access; it produces NOT semantically meaningful vectors. "
        f"Run again with access to the model hub for real results.",
        file=sys.stderr,
    )


def _fallback_vector(key_bytes: bytes) -> np.ndarray:
    """
    Deterministic pseudo-embedding used only when the real CLIP model
    is unavailable. Same input -> same vector, different inputs ->
    (with overwhelming probability) different vectors, and every
    vector is unit-normalized like a real embedding would be -- but
    there is NO semantic relationship between vectors.
    """
    seed = int(hashlib.sha256(key_bytes).hexdigest(), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=EMBEDDING_DIMENSION).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def is_using_fallback() -> bool:
    """Returns True if the offline pseudo-embedder is active."""
    _load_model()
    return _USING_FALLBACK


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of text strings into the shared multimodal vector space."""
    if not texts:
        return []
    _load_model()
    if _USING_FALLBACK:
        return [_fallback_vector(("text::" + t).encode("utf-8")).tolist() for t in texts]

    embeddings = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]


def embed_images(images_b64: List[str]) -> List[List[float]]:
    """Embed a batch of base64-encoded images into the shared multimodal vector space."""
    if not images_b64:
        return []
    _load_model()
    if _USING_FALLBACK:
        return [_fallback_vector(("image::" + b64[:256]).encode("utf-8")).tolist() for b64 in images_b64]

    from PIL import Image

    pil_images = []
    for b64 in images_b64:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        pil_images.append(img)

    embeddings = _model.encode(pil_images, convert_to_numpy=True, show_progress_bar=False)
    return [e.tolist() for e in embeddings]


def embed_image(image_b64: str) -> List[float]:
    return embed_images([image_b64])[0]
