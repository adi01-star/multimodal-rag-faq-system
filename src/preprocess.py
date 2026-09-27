"""
Text Preprocessing Module for Multimodal RAG FAQ System.
Person 2: Preprocessing & Chunking

Provides robust text cleaning, character normalization, and structural preservation
for downstream chunking and retrieval.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize raw text extracted from documents.

    Preserves:
    - Headings, numbered steps (1., 2., Step 1, etc.)
    - Bullet points (*, -, •, etc.)
    - Question/Answer structure (Q1., Q2., Question 1, etc.)
    - URLs (https://, http://, etc.)
    - Meaningful punctuation and casing

    Normalizes:
    - Replaces unicode replacement characters and non-standard dashes/quotes
    - Strips non-printable control characters
    - Normalizes excessive blank lines and horizontal whitespace
    - Strips synthetic boilerplate footers

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned, UTF-8 compatible, normalized text.
    """
    if not text:
        return ""

    # Ensure input is a string
    cleaned = str(text)

    # 1. Normalize unicode characters (NFKC normalization converts compatibility characters)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # 2. Fix specific common PDF extraction artifacts
    # Replacement character \ufffd typically corresponds to en-dash, em-dash, or bullet in PDFs
    cleaned = cleaned.replace("\ufffd", " - ")

    # Normalize smart quotes and special typographical dashes
    smart_replacements = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
        "\u2013": "-",  # En dash
        "\u2014": " - ",  # Em dash
        "\u00a0": " ",  # Non-breaking space
        "\u2022": "* ",  # Bullet
        "\u2023": "* ",  # Triangular bullet
        "\u25e6": "* ",  # White bullet
        "\u2043": "* ",  # Hyphen bullet
    }
    for orig, rep in smart_replacements.items():
        cleaned = cleaned.replace(orig, rep)

    # 3. Remove obvious control characters (preserve newline \n and tab \t)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", cleaned)

    # 4. Remove synthetic boilerplate footers if present (e.g. sample metadata footers)
    # Example: "Source type: Synthetic sample FAQ document | Suitable for local development and extraction testing."
    cleaned = re.sub(
        r"Source\s+type\s*:\s*Synthetic\s+sample\s+FAQ\s+document\s*\|[^\n]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 5. Normalize line-by-line whitespace
    lines = cleaned.splitlines()
    normalized_lines = []
    for line in lines:
        # Strip trailing and leading excessive horizontal whitespace
        # but preserve indentation if line begins with a list marker or step
        l_stripped = line.strip()
        if not l_stripped:
            normalized_lines.append("")
        else:
            # Replace multiple consecutive spaces/tabs within line with a single space
            l_norm = re.sub(r"[ \t]+", " ", l_stripped)
            normalized_lines.append(l_norm)

    cleaned = "\n".join(normalized_lines)

    # 6. Normalize excessive blank lines (collapse 3+ consecutive newlines into 2)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 7. Final strip
    cleaned = cleaned.strip()

    # Ensure UTF-8 compatibility
    cleaned = cleaned.encode("utf-8", "ignore").decode("utf-8")

    return cleaned
