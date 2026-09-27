# Person 3: Embeddings & Vector Database
## Multimodal RAG FAQ System

This is the retrieval backbone: it embeds every text chunk from Person 2
and every image from Person 1 into one shared vector space, indexes
them in a vector database, and gives you a query interface + sample
test queries to check retrieval quality.

---

## 1. Embedding Model

**Model:** `clip-ViT-B-32` (via `sentence-transformers`)
**Dimension:** 512
**Why CLIP:** it's one of the models named in the project brief, it's
well supported, and — crucially — it embeds **both text and images**
into the *same* vector space with an identical `.encode()` call. That
shared space is what lets a text query retrieve an image (e.g. "show
me this error screenshot") and is the whole point of "multimodal" RAG.
Swapping to a domain-fine-tuned CLIP variant later only requires
changing `EMBEDDING_MODEL_NAME` in `person3/embeddings.py`.

### Offline fallback (important)
`person3/embeddings.py` tries to load `clip-ViT-B-32` from the
HuggingFace hub. If that fails (no network / no HF access — as in
some sandboxed dev environments), it **automatically and visibly**
falls back to a deterministic pseudo-embedder so the rest of the
pipeline (chunking → embedding → indexing → retrieval) can still be
built and tested end-to-end. This is clearly logged to stderr and the
fallback vectors are **not semantically meaningful** — they only exist
so nothing else in the pipeline blocks on model-hub access. Run in an
environment with normal internet access (e.g. your own machine / CI)
to get real embeddings; no code changes are needed.

---

## 2. Vector Database

**Choice:** [ChromaDB](https://www.trychroma.com/), embedded/local,
persisted to disk.

**Why Chroma instead of a managed cloud DB (Pinecone/Weaviate/Qdrant Cloud):**
- No API key or hosted account required — anyone on the team (or a
  grader) can reproduce the "live vector database" deliverable by
  just running `run_indexing.py`.
- Same conceptual interface (`add`/`upsert` + `query` with metadata
  filtering) as the managed alternatives, so the indexing/retrieval
  code in `person3/vector_store.py` is a thin, swappable wrapper — if
  you later want Pinecone/Qdrant/Weaviate for production scale, only
  that one file changes.
- Cosine similarity + metadata `where` filters out of the box, which
  covers everything the spec asks for (filter by `category`,
  `source_url`, etc).

### Connection details for this deliverable
| Field | Value |
|---|---|
| Host/Port | none — local embedded database |
| API key | none required |
| Persist directory | `./vector_db/` (created automatically) |
| Text collection ("index") | `faq_text_chunks` |
| Image collection ("index") | `faq_images` |

To point this at a hosted/managed vector DB instead, edit
`person3/vector_store.py` only — `indexer.py` and `test_retrieval.py`
don't need to change since they only call `add_*`/`query_*`.

---

## 3. Input

- `output/chunks.jsonl` — Person 2's chunked text + metadata
  (`chunk_id`, `text`, `parent_doc_id`, `chunk_index`, `metadata.category`,
  `metadata.type`, `metadata.difficulty`, `metadata.language`,
  `metadata.has_image`, `metadata.image_refs`, `metadata.source_url`, ...).
- `input/*.json` — Person 1's per-document JSON files, each containing
  an `images` array with `filename` + base64-encoded PNG data.

Images are deduplicated by filename before embedding (each unique
image is embedded once, even if referenced by multiple chunks), and
each embedded image record tracks which chunk(s) it belongs to via
`parent_chunk` / `parent_chunks`.

---

## 4. Output Format

`run_indexing.py` produces:

1. **A populated local ChromaDB vector database** at `vector_db/`
   (the "live vector database" deliverable), with two collections:
   - `faq_text_chunks`: one vector per text chunk, with metadata
     (`chunk_id`, `parent_doc_id`, `category`, `type`, `difficulty`,
     `language`, `has_image`, `source_url`) for filtered search.
   - `faq_images`: one vector per unique image, with `image_id` and
     `parent_chunks` metadata.

2. **`output/embedding_metadata.jsonl`** — one JSON object per
   embedded item, matching the project's spec format:

   ```json
   {"chunk_id": "chunk_001_v1", "embedding": [0.23, -0.41, ...], "embedding_model": "clip-ViT-B-32", "embedding_dimension": 512, "embedding_type": "text", "indexed_at": "2024-01-15T14:30:00Z"}
   {"image_id": "screenshot_001.png", "embedding": [0.12, -0.33, ...], "embedding_model": "clip-ViT-B-32", "embedding_dimension": 512, "embedding_type": "image", "parent_chunk": "chunk_001_v1", "indexed_at": "2024-01-15T14:35:00Z"}
   ```

3. **`output/embedding_metadata_summary.csv`** — the same records
   without the raw vectors, for quick human inspection
   (`id, embedding_type, embedding_model, embedding_dimension, parent_chunk, indexed_at`).

---

## 5. How to Run

```bash
pip install -r requirements_person3.txt

# 1. Index everything (reads output/chunks.jsonl + input/*.json by default)
python run_indexing.py

# Optional custom paths:
python run_indexing.py --chunks_jsonl output/chunks.jsonl \
                        --input_dir input \
                        --persist_dir vector_db \
                        --output_dir output

# 2. Run sample retrieval test queries against the indexed DB
python test_retrieval.py
```

---

## 6. Sample Test Queries

`test_retrieval.py` embeds five representative FAQ-style questions
(one targeting each of the five sample documents), retrieves the
top-3 nearest text chunks from `faq_text_chunks`, and checks whether
a chunk from the *expected* source document shows up — a simple proxy
for the project's "Top-3 retrieval accuracy > 85%" success criterion:

| Query | Expected source doc |
|---|---|
| "How do I reset my forgotten password?" | `01_account_and_password_faq` |
| "My internet keeps disconnecting, what should I do?" | `02_network_troubleshooting_faq` |
| "Where can I safely download software before installing it?" | `03_software_installation_faq` |
| "How can I tell if a login page is a phishing scam?" | `04_security_and_safe_usage_faq` |
| "What is Retrieval-Augmented Generation and why is it multimodal?" | `05_multimodal_rag_faq_system_faq` |

Add more cases directly to the `SAMPLE_QUERIES` list in
`test_retrieval.py` as the corpus grows past the initial 5-document
sample toward the 500+/2,000+ chunk target.

---

## 7. Scaling to 2,000+ Chunks / 500+ Images

- **Batching:** text chunks are embedded in configurable batches
  (`text_batch_size`, default 32) instead of one call per chunk, to
  keep model calls efficient at scale.
- **Deduplicated image embedding:** each unique image is embedded once
  regardless of how many chunks reference it, avoiding redundant work.
- **Idempotent indexing:** both collections use `upsert`, so re-running
  `run_indexing.py` after Person 2 regenerates `chunks.jsonl` updates
  existing vectors in place instead of duplicating them.
- **Swappable backend:** `person3/vector_store.py` isolates all
  database-specific code, so moving to a managed/cloud vector DB for
  production-scale concurrent querying is a localized change.
