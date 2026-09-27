"""
Complete Multimodal RAG Pipeline
Person 4 + Person 5
"""

from person4.retriever import Retriever
from person5.llm_generator import LLMGenerator


def run_rag(query, top_k=5):

    print("\n" + "=" * 70)
    print("MULTIMODAL RAG FAQ SYSTEM")
    print("=" * 70)

    print("\nUser Question:")
    print(query)

    # =====================================================
    # PERSON 4 - RETRIEVAL
    # =====================================================

    print(
        "\n[Person 4] Retrieving relevant information..."
    )

    retriever = Retriever(
        "vector_db"
    )

    retrieved_results = retriever.retrieve(
        query=query,
        top_k=top_k
    )

    print(
        f"[Person 4] Retrieved "
        f"{len(retrieved_results['text_results'])} "
        f"text results"
    )

    print(
        f"[Person 4] Retrieved "
        f"{len(retrieved_results['image_results'])} "
        f"image results"
    )

    # =====================================================
    # PERSON 5 - MULTIMODAL LLM
    # =====================================================

    print(
        "\n[Person 5] Generating final answer..."
    )

    generator = LLMGenerator()

    result = generator.generate_answer(
        query=query,
        retrieved_results=retrieved_results
    )

    # =====================================================
    # FINAL ANSWER
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(
        "\n" + result["answer"]
    )

    # =====================================================
    # SOURCES
    # =====================================================

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

    # =====================================================
    # IMAGES USED
    # =====================================================

    print(
        "\nImages used by Gemini:"
    )

    if result["images_used"]:

        for image in result["images_used"]:
            print(
                f"- {image}"
            )

    else:

        print(
            "- No images were used."
        )

    print("\n" + "=" * 70)

    return result


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    query = input(
        "\nEnter your FAQ question: "
    )

    run_rag(query)