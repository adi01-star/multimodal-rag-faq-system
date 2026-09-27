# Person 2: Preprocessing & Chunking
## Multimodal RAG FAQ System

This repository contains the complete implementation for **Person 2: Preprocessing & Chunking** of the college final-year project **Multimodal RAG FAQ System**.

The pipeline ingests raw extracted JSON files and images produced by Person 1, performs text normalization, carries out semantic-aware chunking, generates standardized structural metadata, preserves multimodal image references, eliminates exact and near-duplicate chunks, validates all outputs, and prepares retrieval-ready JSONL and CSV deliverables for Person 3's embedding pipeline.

---

## Table of Contents
1. [Project Purpose](#1-project-purpose)
2. [Input Format](#2-input-format)
3. [Preprocessing Steps](#3-preprocessing-steps)
4. [Cleaning Strategy](#4-cleaning-strategy)
5. [Chunking Strategy](#5-chunking-strategy)
6. [Target Chunk Size](#6-target-chunk-size)
7. [Token-Counting Method](#7-token-counting-method)
8. [Metadata Schema](#8-metadata-schema)
9. [Category, Type, and Difficulty Rules](#9-category-type-and-difficulty-rules)
10. [Deduplication Method](#10-deduplication-method)
11. [Image-Reference Handling](#11-image-reference-handling)
12. [Output Files](#12-output-files)
13. [Validation Checks](#13-validation-checks)
14. [How to Run the Pipeline](#14-how-to-run-the-pipeline)
15. [Dependencies](#15-dependencies)
16. [Limitations](#16-limitations)
17. [Scaling to 500+ Documents](#17-scaling-to-500-documents)

---

## 1. Project Purpose
The **Multimodal RAG FAQ System** combines semantic text and visual retrieval to provide accurate, cited answers to user inquiries across technical knowledge domains.

As the second stage of the 5-person late-stage fusion architecture:
- **Upstream Producer (Person 1)**: Extracts raw text, metadata, and embedded images from PDFs/documents into independent JSON files.
- **Our Responsibility (Person 2)**: Transforms raw extracted text and image references into clean, semantically intact, metadata-rich, deduplicated chunks.
- **Downstream Consumer (Person 3)**: Embeds text chunks and images into a vector database (e.g. CLIP / Sentence-BERT + Qdrant / Pinecone).

Person 2 does **not** compute embeddings, build vector indices, or invoke LLMs; our sole focus is data quality, structural integrity, and chunk-to-image alignment.

---

## 2. Input Format
Person 1 supplies extracted documents in an `input/` folder consisting of:
1. Document JSON files (one JSON file per document).
2. An `embedded_images/` folder containing extracted image files (`.png`, `.jpg`).

### Person 1 Document JSON Schema
```json
{
  "id": "01_account_and_password_faq",
  "raw_text": "Account & Password FAQ\nDomain: User account management...\nQ1. How do I reset my password?...",
  "images": [
    {
      "filename": "01_account_and_password_faq_page_1_image_1.png",
      "page": 1,
      "position": [
        {
          "x0": 67.23,
          "y0": 156.00,
          "x1": 528.04,
          "y1": 422.40
        }
      ],
      "base64": "iVBORw0KGgo..."
    }
  ],
  "source_url": null,
  "source_title": "01_account_and_password_faq.pdf",
  "extracted_date": "2026-09-27 11:14:37",
  "author": null
}
```

---

## 3. Preprocessing Steps
The pipeline executes sequentially through the following stages:

```
Person 1 JSON Files & Extracted Images
               ↓
     1. Ingestion & JSON Parsing
               ↓
   2. Text Cleaning & Normalization (`clean_text`)
               ↓
  3. Logical Section & FAQ Structure Detection
               ↓
   4. Semantic-Aware Chunking (250–500 tokens)
               ↓
   5. Metadata Classification & Language Detection
               ↓
  6. Image-Reference Association & Mapping
               ↓
 7. Two-Pass Deduplication (Exact SHA-256 + TF-IDF)
               ↓
  8. Sequential Re-indexing & ID Assignment
               ↓
   9. Output Serialization (`chunks.jsonl`, `chunks_validation.csv`)
               ↓
  10. Automated Validation Suite & Report
```

---

## 4. Cleaning Strategy
Raw extracted text from PDFs frequently contains character corruption, control characters, inconsistent whitespace, and synthetic artifacts. `src/preprocess.py` provides the `clean_text(text: str)` function:

1. **Unicode Normalization (NFKC)**: Converts compatibility glyphs and ligatures into canonical form.
2. **Artifact Correction**: Replaces unmapped replacement characters (`\ufffd`) with standard typographical symbols (`-`).
3. **Smart Character Mapping**: Standardizes curly quotes (`“”`, `‘’`), en/em dashes (`–`, `—`), and non-breaking spaces (`\u00a0`).
4. **Control Character Removal**: Strips ASCII control codes (`\x00-\x08`, `\x0B-\x0C`, `\x0E-\x1F`, `\x7F`) while strictly preserving line breaks (`\n`) and tabs (`\t`).
5. **Boilerplate Stripping**: Removes synthetic generation notices (e.g. `Source type: Synthetic sample FAQ document...`).
6. **Whitespace Normalization**: Normalizes multiple spaces and tabs within lines to single spaces without destroying list indentation.
7. **Paragraph Preservation**: Collapses 3 or more consecutive blank lines into exactly 2 (`\n\n`), preserving clean paragraph blocks.
8. **Structure Preservation**: Preserves question prefixes (`Q1.`, `Q2.`), numbered steps (`1.`, `2.`), bullet points (`*`, `-`), and URLs (`https://`).
9. **UTF-8 Assurance**: Enforces strict UTF-8 string encoding.

---

## 5. Chunking Strategy
To avoid losing semantic coherence or cutting mid-sentence, `src/chunking.py` applies a hierarchical chunking strategy:

1. **Hierarchy Rule 1 (FAQ Atomicity)**: FAQ Questions and Answers (`Q1. ...` + answer) are preserved together as unified chunks.
2. **Hierarchy Rule 2 (Paragraph Preservation)**: Content is split at paragraph boundaries (`\n\n`) whenever possible.
3. **Hierarchy Rule 3 (Instruction Preservation)**: Numbered steps and bulleted lists are kept contiguous within a single step or paragraph.
4. **Hierarchy Rule 4 (Sentence Boundaries)**: Splits occur only on sentence boundaries (`[.!?]\s+`), never cutting sentences in the middle.
5. **Hierarchy Rule 5 (Sub-250 Token Rule)**: If a logical FAQ unit is under 250 tokens, it is kept as one chunk rather than artificially expanded or padded with irrelevant text.
6. **Hierarchy Rule 6 (Super-500 Token Rule)**: If an FAQ or section exceeds 500 tokens, it is recursively split at paragraph, step, and sentence boundaries to stay within 250–500 tokens.

---

## 6. Target Chunk Size
- **Nominal Range**: 250–500 tokens.
- **Handling Short FAQs**: Natural FAQs are frequently concise (30–80 tokens). As instructed, these are kept atomic to maintain high semantic precision for Person 4's retrieval queries.
- **Handling Long Guides**: Multi-step guides exceeding 500 tokens are cleanly segmented into sequential parts (`(Part 1)`, `(Part 2)`) with contextual headers preserved.

---

## 7. Token-Counting Method
Token counting is implemented in `count_tokens(text: str)` in `src/chunking.py`:
- **Primary Method**: `tiktoken` with the `cl100k_base` BPE tokenizer (compatible with OpenAI and modern multimodal embedding models).
- **Offline / Fallback Method**: If `tiktoken` is unavailable or fails, a dual heuristic based on whitespace word count multiplied by 1.3 combined with character length estimation (`len(text) // 4`) is used.
- The resulting value is stored in every chunk under `metadata.tokens_approx`.

---

## 8. Metadata Schema
Every chunk produced conforms strictly to the project specification contract:

```json
{
  "chunk_id": "chunk_001_v1",
  "text": "Open the login page and select Forgot Password...",
  "chunk_number": 2,
  "metadata": {
    "category": "Account Management",
    "type": "howto",
    "difficulty": "beginner",
    "source_doc_id": "01_account_and_password_faq",
    "source_url": null,
    "has_image": true,
    "image_refs": [
      "01_account_and_password_faq_page_1_image_1.png"
    ],
    "language": "en",
    "tokens_approx": 44
  },
  "parent_doc_id": "01_account_and_password_faq",
  "chunk_index": 1
}
```

- `chunk_id`: Globally unique identifier (`chunk_{idx:03d}_v1`).
- `parent_doc_id`: Source document identifier from Person 1.
- `chunk_index`: 0-indexed integer per parent document (`0, 1, 2, ...`).
- `chunk_number`: 1-indexed integer per parent document (`chunk_index + 1`).
- `metadata.tokens_approx`: Integer token estimate.
- `metadata.has_image`: Boolean flag (`True` if `image_refs` is non-empty).
- `metadata.image_refs`: Array of validated image filenames.

---

## 9. Category, Type, and Difficulty Rules

### Controlled Category Vocabulary
Categories are restricted to:
- `Account Management`: Password reset, login, profile, account unlock, credentials.
- `Authentication`: 2FA, MFA, OTP, single sign-on (SSO), token verification.
- `Payments`: Invoices, credit card, subscriptions, billing, refunds.
- `Technical Support`: Software installation, setup, prerequisites, operating systems.
- `Troubleshooting`: Network issues, Wi-Fi errors, connection drops, slow speeds, system crashes.
- `Security`: Phishing, suspicious URLs, account compromise warnings, password strength.
- `General Information`: High-level system architecture, concepts (e.g. Multimodal RAG), overviews.

*Rule: Document-level title and domain are weighted heavily (10x) over isolated word matches (2x) to ensure strong thematic cohesion across all chunks in a document.*

### Controlled Type Vocabulary
- `howto`: Step-by-step procedures, instructions ("How do I...", "To reset...", "Open settings and click...").
- `troubleshooting`: Diagnosing and resolving problems ("Why is my...", "What should I do if...", "No Internet").
- `informational`: Conceptual overviews, architecture explanations, reference tables.
- `faq`: General question-and-answer clarifications.

### Controlled Difficulty Vocabulary
- `beginner`: Routine user tasks (resetting a password, connecting to Wi-Fi, basic account safety).
- `intermediate`: Setup prerequisites, operating system configurations, architecture overviews.
- `advanced`: In-depth debugging, protocol analysis, complex algorithmic workflows.

### Language Detection
Detected using `langdetect` with reliable fallback to `'en'`. Standard two-letter ISO 639-1 code.

---

## 10. Deduplication Method
Preventing duplicate and near-duplicate chunks avoids index bloating and skewed retrieval rankings in Person 3 and Person 4.

Implemented in `src/deduplication.py`:
1. **Exact Duplicate Detection**:
   - Aggressively normalizes text (lowercasing, stripping all punctuation, collapsing whitespace).
   - Computes SHA-256 hash.
   - Any identical chunk encountered after the first is discarded.
2. **Near-Duplicate Detection**:
   - Uses `TfidfVectorizer(ngram_range=(1, 1))` with Cosine Similarity.
   - Compares candidate chunk vector against previously accepted unique chunks.
   - Configurable similarity threshold: `0.88` (default).
   - Chunks exceeding the threshold are logged and discarded as near-duplicates.
3. **Tracking**: The pipeline tracks and reports `total_chunks_before`, `exact_duplicates_removed`, `near_duplicates_removed`, and `final_chunks`.

---

## 11. Image-Reference Handling
Person 2 preserves multimodal links without modifying image files:
1. **Extraction Integrity**: Image files are preserved in `input/embedded_images/`.
2. **Chunk Association**:
   - Chunks explicitly referencing visual elements (`"visual"`, `"diagram"`, `"flowchart"`, `"image"`) are linked to corresponding images.
   - Primary procedural FAQs (e.g. `Q1`) illustrated by the document's page-1 visual flowcharts are linked to that image.
   - Chunks without relevant visual elements have `"has_image": false` and `"image_refs": []`.
3. **Broken Reference Prevention**: Validation verifies that every image filename in `image_refs` exists in Person 1's extracted image set.

---

## 12. Output Files
The pipeline generates deliverables in `output/`:

### A. `output/chunks.jsonl`
- Standard JSON Lines format (one JSON object per line).
- Ready for batch ingestion into Person 3's embedding scripts.

### B. `output/chunks_validation.csv`
Tabular summary for inspection with columns:
`chunk_id`, `doc_id`, `chunk_index`, `chunk_number`, `token_count`, `category`, `type`, `difficulty`, `language`, `has_image`, `image_count`, `source_url`, `text_length`.

---

## 13. Validation Checks
Automated validation is implemented in `src/validation.py` and executed at the end of every pipeline run:
- **JSONL Syntax**: Every line parses as valid JSON.
- **Required Top-Level Keys**: `chunk_id`, `text`, `chunk_number`, `metadata`, `parent_doc_id`, `chunk_index`.
- **Required Metadata Keys**: `category`, `type`, `difficulty`, `source_doc_id`, `source_url`, `has_image`, `image_refs`, `language`, `tokens_approx`.
- **Non-Empty Constraints**: No empty `chunk_id`, no empty `text`.
- **Integrity**: `tokens_approx > 0`, `chunk_number == chunk_index + 1`, `parent_doc_id == metadata.source_doc_id`.
- **Controlled Vocabularies**: Category, type, and difficulty values belong to approved enums.
- **Broken Image References**: 0 broken references permitted.
- **Uniqueness**: 0 duplicate chunk IDs; 0 exact duplicate texts.
- **CSV Parity**: Row count and column values match JSONL exactly.

---

## 14. How to Run the Pipeline

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Execute the Complete Preprocessing Pipeline
```bash
python run_pipeline.py
```

### Custom Arguments (Optional)
```bash
python run_pipeline.py --input_dir input --output_dir output --similarity_threshold 0.88 --target_min 250 --target_max 500
```

### Run Unit Tests
```bash
python test_pipeline.py
```

---

## 15. Dependencies
Specified in `requirements.txt`:
- `tiktoken>=0.7.0`: Token counting using BPE.
- `scikit-learn>=1.3.0`: TF-IDF vectorization and cosine similarity for deduplication.
- `pandas>=2.0.0`: Data handling and CSV validation inspection.
- `langdetect>=1.0.9`: Automatic language identification.

---

## 16. Limitations
1. **Short FAQ Natural Length**: In single-topic FAQs, question and answer pairs are concise (~35–70 tokens). Per Section 5 guidelines, they are preserved intact rather than unnaturally padded.
2. **Single Image Per Page Assumption**: Person 1's test corpus contains 1 visual diagram per document. In a larger corpus with multiple interleaved images, bounding box coordinates (`y0`, `y1`) can be used to link images to the nearest paragraph.

---

## 17. Scaling to 500+ Documents
The codebase is designed for seamless scaling from the 5-document sample to 500+ documents (2,000+ chunks):

1. **Streaming / Iterative Ingestion**: Documents are processed sequentially without loading all raw documents into memory at once.
2. **Sublinear Deduplication**: TF-IDF cosine comparison and SHA-256 hash sets scale efficiently across thousands of chunks in seconds.
3. **Dynamic Re-indexing**: Sequential document and global numbering (`chunk_0001_v1`, `chunk_0002_v1`, etc.) scales automatically.
4. **Validation Throughput**: The validation script parses JSONL line-by-line, validating 5,000+ chunks in under 3 seconds.
5. **Batch Ingestion Ready**: The generated `chunks.jsonl` allows Person 3 to stream chunks into embedding models with standard batch sizing (e.g. batches of 64 or 128).
