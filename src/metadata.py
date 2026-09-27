"""
Metadata Generation and Classification Module.
Person 2: Preprocessing & Chunking

Generates standardized metadata for each chunk:
- category: Account Management, Authentication, Payments, Technical Support, Troubleshooting, Security, General Information
- type: howto, faq, troubleshooting, informational
- difficulty: beginner, intermediate, advanced
- language: ISO 639-1 code (e.g., 'en')
- image_refs: list of valid image filenames mapped to chunk
- tokens_approx: token count
- source_doc_id, source_url
"""

import re
from typing import Dict, Any, List, Optional

try:
    import langdetect
    langdetect.DetectorFactory.seed = 0
except ImportError:
    langdetect = None


ALLOWED_CATEGORIES = {
    "Account Management",
    "Authentication",
    "Payments",
    "Technical Support",
    "Troubleshooting",
    "Security",
    "General Information"
}

ALLOWED_TYPES = {
    "howto",
    "faq",
    "troubleshooting",
    "informational"
}

ALLOWED_DIFFICULTIES = {
    "beginner",
    "intermediate",
    "advanced"
}


def detect_language(text: str) -> str:
    """
    Detect the ISO language code of the text. Defaults to 'en' on failure or ambiguity.
    """
    if not text or len(text.strip()) < 10:
        return "en"

    if langdetect is not None:
        try:
            detected = langdetect.detect(text)
            if detected and len(detected) == 2:
                return detected
        except Exception:
            pass

    return "en"


def classify_category(text: str, doc_metadata: Dict[str, Any], doc_title: str = "") -> str:
    """
    Classify the category using a controlled vocabulary.
    Preserves category if already provided by Person 1.
    Weights document-level domain and title heavily to maintain topic coherence.
    """
    # 1. Preserve if already present and valid
    existing = doc_metadata.get("category")
    if existing and existing in ALLOWED_CATEGORIES:
        return existing

    # Extract domain and title, normalizing underscores to spaces for clean regex matching
    raw_text = doc_metadata.get("raw_text", "")
    domain_match = re.search(r"Domain:\s*([^\n\.]+)", raw_text, re.IGNORECASE)
    domain_str = domain_match.group(1).lower() if domain_match else ""
    domain_str = re.sub(r"[_\-]+", " ", domain_str)

    title_str = re.sub(r"[_\-]+", " ", doc_title.lower())
    chunk_str = re.sub(r"[_\-]+", " ", text.lower())

    # Extract first line of raw_text if it looks like a document title
    first_line = raw_text.splitlines()[0].lower() if raw_text else ""
    doc_heading = re.sub(r"[_\-]+", " ", first_line)

    # Rule-based scoring with keyword matching
    scores = {cat: 0 for cat in ALLOWED_CATEGORIES}

    def score_text(s: str, weight: int):
        if not s:
            return
        if re.search(r"\b(password|reset|account|login|profile|sign in|locked|username)\b", s):
            scores["Account Management"] += weight
        if re.search(r"\b(otp|2fa|mfa|token|authentication|single sign-on|sso)\b", s):
            scores["Authentication"] += weight
        if re.search(r"\b(payment|bill|billing|invoice|refund|credit card|pricing|subscription)\b", s):
            scores["Payments"] += weight
        if re.search(r"\b(network|wifi|wi-fi|internet|connect|connection|slow|router|offline|disconnect|troubleshoot)\b", s):
            scores["Troubleshooting"] += weight
        if re.search(r"\b(security|phishing|suspicious|safe|safety|malware|threat|attack|unauthorized|https|warning)\b", s):
            scores["Security"] += weight
        if re.search(r"\b(install|installation|setup|software|prerequisite|download|installer|configure)\b", s):
            scores["Technical Support"] += weight
        if re.search(r"\b(rag|retrieval|multimodal|language model|llm|generation|embedding|technical|citation)\b", s):
            scores["General Information"] += weight

    # Document title, heading, and domain establish document-level topic
    score_text(title_str, weight=10)
    score_text(doc_heading, weight=10)
    score_text(domain_str, weight=10)

    # Chunk-specific content gets secondary weight
    score_text(chunk_str, weight=2)

    # Pick the category with the highest positive score
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] > 0:
        return best_cat

    return "General Information"


def classify_type(chunk_text: str, section_type: str = "") -> str:
    """
    Classify the chunk type: howto, faq, troubleshooting, informational.
    """
    lower = chunk_text.lower()

    if section_type == "overview" or "dataset purpose:" in lower or "reference information" in lower:
        return "informational"

    # Conceptual/architectural questions
    if re.search(r"\b(what is retrieval|why does this project|why are citations|what should the system do)\b", lower):
        return "informational"

    # Procedural How-To questions
    if re.search(r"\b(how do i|how can i|to reset|step \d+|1\.\s+open|follow the reset link|open account settings)\b", lower):
        return "howto"

    # Troubleshooting problems / errors / failures
    if re.search(r"\b(error message|troubleshoot|slow|no internet|locked|stops with an error|does not start)\b", lower):
        return "troubleshooting"
    if re.search(r"\bwhy is my\b", lower):
        return "troubleshooting"

    # FAQ questions
    if re.search(r"\b(what|why|where|can i|should i|q\d+[\.\:])\b", lower):
        return "faq"

    return "informational"


def classify_difficulty(chunk_text: str, category: str, chunk_type: str) -> str:
    """
    Classify difficulty: beginner, intermediate, advanced.
    """
    lower = chunk_text.lower()

    # Advanced indicators
    if re.search(r"\b(architecture|encoder|cross-modality|vector database|bm25|re-ranking|multimodal rag|api|parameter)\b", lower):
        return "intermediate" if len(chunk_text) < 400 else "advanced"

    # Intermediate indicators
    if re.search(r"\b(configuration|prerequisite|storage|operating system|permissions|system requirements)\b", lower):
        return "intermediate"

    # Standard user questions are beginner-friendly
    return "beginner"


def map_chunk_images(
    chunk_text: str,
    chunk_index: int,
    section_type: str,
    doc_images: List[Dict[str, Any]]
) -> List[str]:
    """
    Determines which images from Person 1's document belong to this chunk.

    Rules:
    1. If no images in document -> returns []
    2. If chunk text explicitly mentions 'visual', 'diagram', 'screenshot', 'flowchart', 'image' -> link image
    3. If chunk is the first procedural step / primary question (e.g. Q1) or Overview on page 1 -> link image
    4. Image filenames are verified against Person 1's actual extracted image list.
    """
    if not doc_images:
        return []

    available_filenames = [img.get("filename") for img in doc_images if img.get("filename")]
    if not available_filenames:
        return []

    lower = chunk_text.lower()
    matched_images = []

    # Rule 1: Text directly mentions visual or image
    has_visual_mention = bool(re.search(r"\b(visual|diagram|screenshot|flowchart|flow|image|figure)\b", lower))
    if has_visual_mention:
        for fname in available_filenames:
            if fname not in matched_images:
                matched_images.append(fname)
        return matched_images

    # Rule 2: Primary procedural FAQ (Q1 / chunk_index 1) directly linked to page-1 flowcharts
    # In Person 1's dataset, each document's page 1 visual directly depicts Q1's topic
    if chunk_index == 1 and section_type == "faq_item":
        for img in doc_images:
            fname = img.get("filename")
            if fname and fname not in matched_images:
                matched_images.append(fname)

    return matched_images


def build_chunk_metadata(
    chunk_text: str,
    chunk_index: int,
    section_type: str,
    doc_id: str,
    source_url: Optional[str],
    doc_title: str,
    doc_images: List[Dict[str, Any]],
    tokens_approx: int,
    raw_doc_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build the complete metadata dictionary adhering to Person 2 specification.
    """
    category = classify_category(chunk_text, raw_doc_data, doc_title)
    ctype = classify_type(chunk_text, section_type)
    difficulty = classify_difficulty(chunk_text, category, ctype)
    language = detect_language(chunk_text)
    image_refs = map_chunk_images(chunk_text, chunk_index, section_type, doc_images)

    metadata = {
        "category": category,
        "type": ctype,
        "difficulty": difficulty,
        "source_doc_id": doc_id,
        "source_url": source_url,
        "has_image": len(image_refs) > 0,
        "image_refs": image_refs,
        "language": language,
        "tokens_approx": tokens_approx
    }

    return metadata
