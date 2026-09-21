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


def get_graph_relationships(graph, scheme_url):

    relationships = []

    for scheme_node, scheme_data in graph.nodes(data=True):

        if scheme_data.get("node_type") != "scheme":
            continue

        if scheme_data.get("url") != scheme_url:
            continue

        for _, target_node, edge_data in graph.out_edges(
            scheme_node,
            data=True
        ):

            target_data = graph.nodes[target_node]

            relationships.append(
                {
                    "relationship": edge_data.get("relationship"),
                    "target": target_data.get("name")
                }
            )

        break

    return relationships


def build_hybrid_context(
    vector_documents,
    graph,
    graph_schemes
):

    # --------------------------------------------------
    # Store schemes using URL as unique identifier
    # --------------------------------------------------

    schemes = {}

    # --------------------------------------------------
    # Add Vector results
    # --------------------------------------------------

    for document in vector_documents:

        url = document.metadata.get("url")

        if not url:
            continue

        schemes[url] = {
            "title": document.metadata.get("title"),
            "url": url,
            "content": document.page_content
        }

    # --------------------------------------------------
    # Add Graph results
    # --------------------------------------------------

    for scheme in graph_schemes:

        url = scheme.get("url")

        if not url:
            continue

        # If scheme already came from Vector Search,
        # keep the vector content and add graph information.
        if url in schemes:

            schemes[url]["graph_relationships"] = (
                get_graph_relationships(
                    graph,
                    url
                )
            )

        # If scheme only came from Graph Search,
        # create a new entry.
        else:

            schemes[url] = {
                "title": scheme.get("title"),
                "url": url,
                "content": f"""
Scheme Title: {scheme.get("title")}

Description:
{scheme.get("description")}

Funding Pattern:
{scheme.get("funding_pattern")}

How To Avail:
{scheme.get("how_to_avail")}
""".strip(),
                "graph_relationships": (
                    get_graph_relationships(
                        graph,
                        url
                    )
                )
            }

    return schemes


def main():

    load_dotenv()

    question = "Which schemes provide subsidies to farmers?"

    # ==================================================
    # VECTOR RETRIEVAL
    # ==================================================

    print("=" * 70)
    print("1. VECTOR RETRIEVAL")
    print("=" * 70)

    vectorstore = load_vectorstore()

    vector_documents = vectorstore.similarity_search(
        question,
        k=5
    )

    print(
        "Vector documents:",
        len(vector_documents)
    )

    # ==================================================
    # GRAPH RETRIEVAL
    # ==================================================

    print("\n")
    print("=" * 70)
    print("2. GRAPH RETRIEVAL")
    print("=" * 70)

    graph = load_graph()

    graph_schemes = graph_search(graph)

    print(
        "Graph schemes:",
        len(graph_schemes)
    )

    # ==================================================
    # BUILD HYBRID CONTEXT
    # ==================================================

    print("\n")
    print("=" * 70)
    print("3. BUILDING HYBRID CONTEXT")
    print("=" * 70)

    schemes = build_hybrid_context(
        vector_documents,
        graph,
        graph_schemes
    )

    print(
        "Unique schemes:",
        len(schemes)
    )

    # ==================================================
    # DISPLAY HYBRID CONTEXT
    # ==================================================

    print("\n")
    print("=" * 70)
    print("4. FINAL HYBRID CONTEXT")
    print("=" * 70)

    for index, scheme in enumerate(
        schemes.values(),
        start=1
    ):

        print("\n")
        print("-" * 70)

        print(
            f"SCHEME {index}: "
            f"{scheme['title']}"
        )

        print(
            "\nURL:",
            scheme["url"]
        )

        print(
            "\nCONTENT:"
        )

        print(
            scheme["content"]
        )

        print(
            "\nGRAPH RELATIONSHIPS:"
        )

        relationships = scheme.get(
            "graph_relationships",
            []
        )

        if relationships:

            for relationship in relationships:

                print(
                    f"  "
                    f"{relationship['relationship']}"
                    f" -> "
                    f"{relationship['target']}"
                )

        else:

            print("  No graph relationships")

    print("\n")
    print("=" * 70)
    print("HYBRID CONTEXT BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()