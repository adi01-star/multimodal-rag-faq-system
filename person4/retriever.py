"""
Person 4: Retrieval and Ranking

Responsibilities:
1. Convert user query into CLIP embedding
2. Retrieve relevant text chunks from ChromaDB
3. Retrieve relevant images from ChromaDB
4. Rank results using similarity scores
5. Return the best context for Person 5 (LLM)
"""

from person3.embeddings import embed_text
from person3.vector_store import VectorStore


class Retriever:

    def __init__(self, persist_directory="vector_db"):
        self.store = VectorStore(persist_directory)

    def retrieve(self, query, top_k=5):
        """
        Retrieve and rank relevant text chunks and images.
        """

        # 1. Convert query into embedding
        query_embedding = embed_text(query)

        # 2. Retrieve text
        text_results = self.store.query_text(
            query_embedding,
            top_k=top_k
        )

        # 3. Retrieve images
        image_results = self.store.query_images(
            query_embedding,
            top_k=top_k
        )

        # 4. Rank text
        ranked_text = self._process_text_results(
            text_results
        )

        # 5. Rank images
        ranked_images = self._process_image_results(
            image_results
        )

        return {
            "query": query,
            "text_results": ranked_text,
            "image_results": ranked_images
        }

    def _process_text_results(self, results):
        """
        Process and rank text results.
        """

        ranked_results = []

        if not results or not results.get("ids"):
            return ranked_results

        ids = results["ids"][0]

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        for i in range(len(ids)):

            distance = (
                distances[i]
                if i < len(distances)
                else None
            )

            if distance is not None:
                relevance_score = 1 / (1 + distance)
            else:
                relevance_score = 0

            ranked_results.append({
                "rank": i + 1,
                "chunk_id": ids[i],
                "text": (
                    documents[i]
                    if i < len(documents)
                    else ""
                ),
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas)
                    else {}
                ),
                "distance": distance,
                "relevance_score": relevance_score
            })

        # Highest score first
        ranked_results.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        # Re-number ranks
        for rank, item in enumerate(
            ranked_results,
            start=1
        ):
            item["rank"] = rank

        return ranked_results

    def _process_image_results(self, results):
        """
        Process and rank image results.

        Only the top 2 most relevant images
        are returned to Person 5.
        """

        ranked_results = []

        if not results or not results.get("ids"):
            return ranked_results

        ids = results["ids"][0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        for i in range(len(ids)):

            distance = (
                distances[i]
                if i < len(distances)
                else None
            )

            if distance is not None:
                relevance_score = 1 / (1 + distance)
            else:
                relevance_score = 0

            ranked_results.append({
                "rank": i + 1,
                "image_id": ids[i],
                "metadata": (
                    metadatas[i]
                    if i < len(metadatas)
                    else {}
                ),
                "distance": distance,
                "relevance_score": relevance_score
            })

        # Highest score first
        ranked_results.sort(
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        # Re-number ranks
        for rank, item in enumerate(
            ranked_results,
            start=1
        ):
            item["rank"] = rank

        # IMPORTANT:
        # Send only top 2 images to Person 5
        return ranked_results[:2]


def retrieve(
    query,
    top_k=5,
    persist_directory="vector_db"
):
    """
    Simple function for other modules.
    """

    retriever = Retriever(
        persist_directory
    )

    return retriever.retrieve(
        query=query,
        top_k=top_k
    )


if __name__ == "__main__":

    query = "How do I reset my password?"

    results = retrieve(query)

    print("\n" + "=" * 60)
    print("PERSON 4 - RETRIEVAL AND RANKING TEST")
    print("=" * 60)

    print(
        f"\nQuery: {results['query']}"
    )

    print("\n--- Ranked Text Results ---")

    for result in results["text_results"]:

        print(
            f"\nRank: {result['rank']}"
            f"\nChunk ID: {result['chunk_id']}"
            f"\nScore: "
            f"{result['relevance_score']:.4f}"
            f"\nSource: "
            f"{result['metadata'].get('parent_doc_id')}"
        )

        print(
            f"Text: {result['text'][:300]}..."
        )

    print(
        "\n--- Top 2 Ranked Image Results ---"
    )

    for result in results["image_results"]:

        print(
            f"\nRank: {result['rank']}"
            f"\nImage ID: {result['image_id']}"
            f"\nScore: "
            f"{result['relevance_score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("PERSON 4 TEST COMPLETE")
    print("=" * 60)