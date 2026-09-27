"""
Person 5: Multimodal LLM Answer Generation

Responsibilities:
1. Receive retrieved text and images from Person 4
2. Load actual retrieved image files
3. Send text + images to Gemini
4. Generate final answer
5. Return source information and images used
"""

import os
from pathlib import Path

from google import genai
from google.genai import types


class LLMGenerator:

    def __init__(self):

        # -------------------------------------------------
        # Gemini API
        # -------------------------------------------------

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable "
                "is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = "gemini-3.8-flash"

        # -------------------------------------------------
        # Image folder
        # -------------------------------------------------

        self.image_folder = (
            Path(__file__).resolve().parent.parent
            / "input"
            / "embedded_images"
            / "extracted_images"
        )

    # =====================================================
    # FIND IMAGE
    # =====================================================

    def _find_image(self, image_id):

        if not image_id:
            return None

        # Direct path
        image_path = (
            self.image_folder / image_id
        )

        if image_path.exists():
            return image_path

        # Recursive search
        matches = list(
            self.image_folder.rglob(
                image_id
            )
        )

        if matches:
            return matches[0]

        return None

    # =====================================================
    # GENERATE ANSWER
    # =====================================================

    def generate_answer(
        self,
        query,
        retrieved_results
    ):

        # -------------------------------------------------
        # TEXT CONTEXT
        # -------------------------------------------------

        text_context = []

        for item in retrieved_results.get(
            "text_results",
            []
        ):

            text = item.get(
                "text",
                ""
            )

            # Person 4 stores source inside metadata
            metadata = item.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "parent_doc_id",
                item.get(
                    "source",
                    "Unknown"
                )
            )

            score = item.get(
                "relevance_score",
                item.get(
                    "score",
                    0
                )
            )

            text_context.append(
                f"""
Source: {source}
Relevance Score: {score:.4f}

Content:
{text}
"""
            )

        combined_text = "\n".join(
            text_context
        )

        # -------------------------------------------------
        # LOAD ACTUAL IMAGES
        # -------------------------------------------------

        image_parts = []

        image_sources = []

        for item in retrieved_results.get(
            "image_results",
            []
        ):

            image_id = (
                item.get("image_id")
                or item.get("id")
                or item.get("image")
            )

            if not image_id:
                continue

            image_path = self._find_image(
                image_id
            )

            if image_path is None:
                print(
                    f"[WARNING] Image not found: "
                    f"{image_id}"
                )
                continue

            try:

                with open(
                    image_path,
                    "rb"
                ) as file:

                    image_bytes = file.read()

                image_parts.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png"
                    )
                )

                image_sources.append(
                    image_path.name
                )

                print(
                    f"[Person 5] Loaded image: "
                    f"{image_path.name}"
                )

            except Exception as e:

                print(
                    f"[WARNING] Could not load image "
                    f"{image_path.name}: {e}"
                )

        # -------------------------------------------------
        # MULTIMODAL PROMPT
        # -------------------------------------------------

        prompt = f"""
You are an FAQ assistant for a Multimodal RAG system.

Answer the user's question using ONLY the retrieved
text and retrieved images.

USER QUESTION:
{query}

RETRIEVED TEXT:
{combined_text}

The images provided after this prompt are retrieved
from the FAQ knowledge base.

IMPORTANT RULES:

1. Use the retrieved text and actual images as evidence.
2. Examine the images carefully.
3. Use information from diagrams, screenshots,
   flowcharts, or warnings when relevant.
4. Do not invent information.
5. Give a clear and simple answer.
6. Use numbered steps when appropriate.
7. Keep the answer concise and useful.
"""

        # -------------------------------------------------
        # SEND TEXT + IMAGES TO GEMINI
        # -------------------------------------------------

        print(
            "\n[Person 5] Sending retrieved text "
            "and actual images to Gemini..."
        )

        contents = [prompt]

        # Add actual image data
        contents.extend(
            image_parts
        )

        response = (
            self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
        )

        answer = response.text.strip()

        # -------------------------------------------------
        # COLLECT SOURCES
        # -------------------------------------------------

        sources = []

        for item in retrieved_results.get(
            "text_results",
            []
        ):

            metadata = item.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "parent_doc_id",
                item.get("source")
            )

            if (
                source
                and source not in sources
            ):
                sources.append(
                    source
                )

        # -------------------------------------------------
        # RETURN COMPLETE RESULT
        # -------------------------------------------------

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "image_results": retrieved_results.get(
                "image_results",
                []
            ),
            "images_used": image_sources
        }


# =========================================================
# STANDALONE PERSON 5 TEST
# =========================================================

if __name__ == "__main__":

    from person4.retriever import Retriever

    print("=" * 70)
    print(
        "PERSON 5 - MULTIMODAL LLM ANSWER "
        "GENERATION TEST"
    )
    print("=" * 70)

    query = input(
        "\nEnter your FAQ question: "
    )

    # -----------------------------------------------------
    # Person 4
    # -----------------------------------------------------

    print(
        "\n[Person 4] Retrieving text and images..."
    )

    retriever = Retriever(
        "vector_db"
    )

    retrieved_results = retriever.retrieve(
        query=query,
        top_k=5
    )

    print(
        f"Text results: "
        f"{len(retrieved_results['text_results'])}"
    )

    print(
        f"Image results: "
        f"{len(retrieved_results['image_results'])}"
    )

    # -----------------------------------------------------
    # Person 5
    # -----------------------------------------------------

    print(
        "\n[Person 5] Sending retrieved text "
        "and actual images to Gemini..."
    )

    generator = LLMGenerator()

    result = generator.generate_answer(
        query=query,
        retrieved_results=retrieved_results
    )

    # -----------------------------------------------------
    # FINAL ANSWER
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        "\n" + result["answer"]
    )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    print("\nSources:")

    if result["sources"]:

        for source in result["sources"]:
            print(
                f"- {source}"
            )

    else:

        print(
            "- No source documents found."
        )

    # -----------------------------------------------------
    # IMAGES USED
    # -----------------------------------------------------

    print(
        "\nImages actually used by Gemini:"
    )

    if result["images_used"]:

        for image in result["images_used"]:
            print(
                f"- {image}"
            )

    else:

        print(
            "- No images were loaded."
        )

    print("\n" + "=" * 70)
    print(
        "PERSON 5 MULTIMODAL TEST COMPLETE"
    )
    print("=" * 70)