import pickle

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_DIR = "vectorstore"
GRAPH_FILE = "data/raw/knowledge_graph.gpickle"


def load_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def load_graph():

    with open(GRAPH_FILE, "rb") as file:
        return pickle.load(file)


def graph_search(graph):

    results = []

    for scheme_node, scheme_data in graph.nodes(data=True):

        if scheme_data.get("node_type") != "scheme":
            continue

        has_farmers = False
        has_subsidy = False

        for _, target_node, edge_data in graph.out_edges(
            scheme_node,
            data=True
        ):

            relationship = edge_data.get("relationship")
            target_data = graph.nodes[target_node]

            if (
                relationship == "BENEFITS"
                and target_data.get("name") == "Farmers"
            ):
                has_farmers = True

            if (
                relationship == "HAS_BENEFIT_TYPE"
                and target_data.get("name") == "Subsidy"
            ):
                has_subsidy = True

        if has_farmers and has_subsidy:
            results.append(scheme_data)

    return results


def main():

    load_dotenv()

    question = "Which schemes provide subsidies to farmers?"

    # ==================================================
    # 1. VECTOR RETRIEVAL
    # ==================================================

    print("=" * 70)
    print("VECTOR RETRIEVAL")
    print("=" * 70)

    vectorstore = load_vectorstore()

    vector_results = vectorstore.similarity_search(
        question,
        k=5
    )

    print(f"\nRetrieved {len(vector_results)} vector documents.")

    # ==================================================
    # 2. GRAPH RETRIEVAL
    # ==================================================

    print("\n")
    print("=" * 70)
    print("GRAPH RETRIEVAL")
    print("=" * 70)

    graph = load_graph()

    graph_results = graph_search(graph)

    print(
        f"\nRetrieved {len(graph_results)} graph schemes."
    )

    # ==================================================
    # 3. COMBINE RESULTS
    # ==================================================

    print("\n")
    print("=" * 70)
    print("COMBINED HYBRID CONTEXT")
    print("=" * 70)

    combined_context = []

    # --------------------------------------------------
    # Add Vector Results
    # --------------------------------------------------

    print("\n--- VECTOR RESULTS ---")

    for index, document in enumerate(
        vector_results,
        start=1
    ):

        title = document.metadata.get("title")

        print(f"\nVector Result {index}: {title}")

        combined_context.append(
            {
                "source": "vector",
                "title": title,
                "content": document.page_content
            }
        )

    # --------------------------------------------------
    # Add Graph Results
    # --------------------------------------------------

    print("\n\n--- GRAPH RESULTS ---")

    for index, scheme in enumerate(
        graph_results,
        start=1
    ):

        title = scheme.get("title")

        print(
            f"\nGraph Result {index}: {title}"
        )

        graph_context = f"""
Scheme Title: {scheme.get("title")}

URL: {scheme.get("url")}

Description:
{scheme.get("description")}

Funding Pattern:
{scheme.get("funding_pattern")}

How To Avail:
{scheme.get("how_to_avail")}
""".strip()

        combined_context.append(
            {
                "source": "graph",
                "title": title,
                "content": graph_context
            }
        )

    # ==================================================
    # 4. SHOW FINAL CONTEXT SIZE
    # ==================================================

    print("\n")
    print("=" * 70)
    print("HYBRID RETRIEVAL SUMMARY")
    print("=" * 70)

    print(
        "\nVector results:",
        len(vector_results)
    )

    print(
        "Graph results:",
        len(graph_results)
    )

    print(
        "Total combined results:",
        len(combined_context)
    )


if __name__ == "__main__":
    main()