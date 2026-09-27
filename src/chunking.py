"""
Chunking Module for Multimodal RAG FAQ System.
Person 2: Preprocessing & Chunking

Implements semantic-aware chunking preserving:
1. FAQ question + answer pairs
2. Paragraphs and structured lists
3. Sentence boundaries
Target chunk size: 250-500 tokens (with preservation of atomic FAQ units under 250 tokens).
Includes robust token counting using tiktoken with fallback.
"""

import re
from typing import List, Dict, Any, Optional

# Attempt to load tiktoken
try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODER = None


def count_tokens(text: str) -> int:
    """
    Estimate the number of tokens in the given text.
    Uses tiktoken (cl100k_base) if available, with a reliable fallback
    based on whitespace/punctuation splitting.

    Args:
        text (str): Input text.

    Returns:
        int: Approximate token count (tokens_approx).
    """
    if not text:
        return 0

    if _TIKTOKEN_ENCODER is not None:
        try:
            return len(_TIKTOKEN_ENCODER.encode(text))
        except Exception:
            pass

    # Reliable fallback: standard English text averages ~1.3 tokens per whitespace word,
    # plus count of punctuation tokens.
    words = text.split()
    if not words:
        return 0
    # Rule of thumb: ~4 characters per token or 1.3 words per token
    char_estimate = max(1, len(text) // 4)
    word_estimate = max(1, int(len(words) * 1.3))
    return int((char_estimate + word_estimate) / 2)


def split_large_section(section_text: str, target_min: int = 250, target_max: int = 500) -> List[str]:
    """
    Intelligently splits a large section (> target_max tokens) into semantic sub-chunks.

    Hierarchy:
    1. Paragraph boundaries (\\n\\n)
    2. Numbered steps / bullet points (1., 2., *, -, Step N)
    3. Sentence boundaries (. ! ?)
    Avoids cutting sentences in the middle.

    Args:
        section_text (str): Large text block to split.
        target_min (int): Minimum target tokens per sub-chunk.
        target_max (int): Maximum target tokens per sub-chunk.

    Returns:
        List[str]: List of sub-chunk texts.
    """
    tokens = count_tokens(section_text)
    if tokens <= target_max:
        return [section_text]

    # Attempt 1: Split by paragraphs
    paragraphs = [p.strip() for p in section_text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        chunks = []
        current_chunk = []
        current_tokens = 0

        for p in paragraphs:
            p_tok = count_tokens(p)
            if current_tokens + p_tok <= target_max:
                current_chunk.append(p)
                current_tokens += p_tok
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                if p_tok > target_max:
                    # Paragraph itself is too large, split by sentences/steps
                    sub_chunks = _split_by_sentences_or_steps(p, target_min, target_max)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_tokens = 0
                else:
                    current_chunk = [p]
                    current_tokens = p_tok

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    # Fallback to sentence/step splitting
    return _split_by_sentences_or_steps(section_text, target_min, target_max)


def _split_by_sentences_or_steps(text: str, target_min: int, target_max: int) -> List[str]:
    """
    Split text by numbered steps, bullet points, or sentence boundaries.
    """
    # Regex matching sentence boundaries or step beginnings
    # Matches period/exclamation/question followed by space, or newline before numbered step
    pattern = r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=\n)(?=\d+[\.\)]|\*|-|Step\s+\d+:)"
    parts = re.split(pattern, text)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [text]

    chunks = []
    current_parts = []
    current_tokens = 0

    for part in parts:
        p_tok = count_tokens(part)
        if current_tokens + p_tok <= target_max:
            current_parts.append(part)
            current_tokens += p_tok
        else:
            if current_parts:
                chunks.append(" ".join(current_parts))
            current_parts = [part]
            current_tokens = p_tok

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def detect_logical_sections(doc_text: str) -> List[Dict[str, Any]]:
    """
    Detects logical sections within a document.

    Recognizes:
    - Document Overview / Header intro
    - FAQ Question & Answer pairs (e.g. Q1., Q2., Question 1, etc.)
    - Reference Information / Tables

    Returns a list of dicts: [{'title': str, 'text': str, 'section_type': str}]
    """
    sections = []

    # Pattern to identify start of FAQ questions
    # e.g., "Q1.", "Q2.", "Question 1:", "1.", etc.
    faq_pattern = r"(?:^|\n)(?=Q\d+[\.\:]|Question\s+\d+[\.\:])"
    parts = re.split(faq_pattern, doc_text)

    if len(parts) > 1:
        # Part 0 is document overview / intro before the first question
        intro = parts[0].strip()
        if intro:
            # Check if intro contains title or domain
            sections.append({
                "title": intro.splitlines()[0] if intro else "Overview",
                "text": intro,
                "section_type": "overview"
            })

        for p in parts[1:]:
            p_strip = p.strip()
            if not p_strip:
                continue

            # Check if there is a trailing reference section or table attached
            ref_pattern = r"(?:^|\n)(?=Reference\s+Information|Appendix|Table\s+\d+:)"
            sub_parts = re.split(ref_pattern, p_strip, maxsplit=1)

            # The FAQ item itself
            faq_text = sub_parts[0].strip()
            if faq_text:
                q_match = re.match(r"(Q\d+[\.\:].*?)(?:\n|$)", faq_text)
                title = q_match.group(1).strip() if q_match else faq_text.splitlines()[0]
                sections.append({
                    "title": title,
                    "text": faq_text,
                    "section_type": "faq_item"
                })

            # Any trailing reference info
            if len(sub_parts) > 1 and sub_parts[1].strip():
                ref_text = sub_parts[1].strip()
                sections.append({
                    "title": ref_text.splitlines()[0],
                    "text": ref_text,
                    "section_type": "reference"
                })
    else:
        # Document without explicit Q1./Q2. markers: split by section headers or large paragraphs
        # Look for Markdown headers (## ) or capitalized standalone headings
        heading_pattern = r"(?:^|\n)(?=#{1,3}\s+[^\n]+|[A-Z][A-Za-z0-9\s]{3,40}:)"
        heading_parts = re.split(heading_pattern, doc_text)
        if len(heading_parts) > 1:
            for hp in heading_parts:
                h_strip = hp.strip()
                if h_strip:
                    sections.append({
                        "title": h_strip.splitlines()[0],
                        "text": h_strip,
                        "section_type": "section"
                    })
        else:
            # Entire text is a single section
            sections.append({
                "title": doc_text.splitlines()[0] if doc_text else "Document",
                "text": doc_text,
                "section_type": "document"
            })

    return sections


def chunk_document(doc_text: str, target_min: int = 250, target_max: int = 500) -> List[Dict[str, Any]]:
    """
    Applies semantic chunking strategy to a cleaned document text.

    - Detects logical FAQ and structural sections.
    - Keeps FAQ Question + Answer together.
    - If a logical FAQ section is < target_min tokens, keeps it as one chunk (avoids artificial expansion).
    - If a section is > target_max tokens, splits it hierarchically.
    - Calculates tokens_approx for each chunk.

    Returns:
        List[Dict[str, Any]]: List of chunk objects with 'text', 'title', 'section_type', 'tokens_approx'.
    """
    sections = detect_logical_sections(doc_text)
    raw_chunks = []

    for sec in sections:
        sec_text = sec["text"]
        tok_count = count_tokens(sec_text)

        if tok_count <= target_max:
            # Under target_max: keep as single atomic chunk
            raw_chunks.append({
                "title": sec["title"],
                "text": sec_text,
                "section_type": sec["section_type"],
                "tokens_approx": tok_count
            })
        else:
            # Large section: split intelligently
            sub_texts = split_large_section(sec_text, target_min, target_max)
            for i, st in enumerate(sub_texts):
                raw_chunks.append({
                    "title": f"{sec['title']} (Part {i+1})",
                    "text": st,
                    "section_type": sec["section_type"],
                    "tokens_approx": count_tokens(st)
                })

    return raw_chunks
