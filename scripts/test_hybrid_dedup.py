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

    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


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
    # 1. VECTOR SEARCH
    # ==================================================

    vectorstore = load_vectorstore()

    vector_documents = vectorstore.similarity_search(
        question,
        k=5
    )

    # Convert vector results into a dictionary using URL
    vector_results = {}

    for document in vector_documents:

        url = document.metadata.get("url")

        if url:
            vector_results[url] = document

    # ==================================================
    # 2. GRAPH SEARCH
    # ==================================================

    graph = load_graph()

    graph_schemes = graph_search(graph)

    # Convert graph results into a dictionary using URL
    graph_results = {}

    for scheme in graph_schemes:

        url = scheme.get("url")

        if url:
            graph_results[url] = scheme

    # ==================================================
    # 3. FIND OVERLAPPING RESULTS
    # ==================================================

    vector_urls = set(vector_results.keys())
    graph_urls = set(graph_results.keys())

    both = vector_urls.intersection(graph_urls)

    vector_only = vector_urls - graph_urls
    graph_only = graph_urls - vector_urls

    # ==================================================
    # 4. DISPLAY RESULTS
    # ==================================================

    print("=" * 70)
    print("HYBRID RETRIEVAL DEDUPLICATION")
    print("=" * 70)

    print("\nQuestion:")
    print(question)

    print("\n")
    print("=" * 70)
    print("RESULT COUNTS")
    print("=" * 70)

    print("\nVector results :", len(vector_results))
    print("Graph results  :", len(graph_results))
    print("Found in both  :", len(both))
    print("Vector only    :", len(vector_only))
    print("Graph only     :", len(graph_only))

    # --------------------------------------------------
    # Results found by both
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FOUND BY BOTH VECTOR + GRAPH")
    print("=" * 70)

    for url in both:

        title = vector_results[url].metadata.get("title")

        print("\n-", title)

    # --------------------------------------------------
    # Vector only
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VECTOR ONLY")
    print("=" * 70)

    for url in vector_only:

        title = vector_results[url].metadata.get("title")

        print("\n-", title)

    # --------------------------------------------------
    # Graph only
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GRAPH ONLY")
    print("=" * 70)

    for url in graph_only:

        title = graph_results[url].get("title")

        print("\n-", title)

    # ==================================================
    # 5. CREATE FINAL UNIQUE SET
    # ==================================================

    all_urls = vector_urls.union(graph_urls)

    print("\n")
    print("=" * 70)
    print("FINAL HYBRID RESULT")
    print("=" * 70)

    print(
        "\nUnique schemes after deduplication:",
        len(all_urls)
    )

    for index, url in enumerate(all_urls, start=1):

        if url in vector_results:

            title = vector_results[url].metadata.get("title")

        else:

            title = graph_results[url].get("title")

        print(f"{index}. {title}")


if __name__ == "__main__":
    main()