import json
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

    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(question)

    # --------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("1. VECTOR SEARCH")
    print("=" * 70)

    vectorstore = load_vectorstore()

    vector_results = vectorstore.similarity_search(
        question,
        k=5
    )

    print(f"\nVector Search returned: {len(vector_results)} documents")

    for index, document in enumerate(vector_results, start=1):

        print(f"\n{index}. {document.metadata.get('title')}")

        print(
            "   Department:",
            document.metadata.get("department")
        )

        print(
            "   Beneficiary:",
            document.metadata.get("beneficiaries")
        )

        print(
            "   Benefit Type:",
            document.metadata.get("benefit_type")
        )

    # --------------------------------------------------
    # KNOWLEDGE GRAPH SEARCH
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("2. KNOWLEDGE GRAPH SEARCH")
    print("=" * 70)

    graph = load_graph()

    graph_results = graph_search(graph)

    print(
        f"\nKnowledge Graph returned: "
        f"{len(graph_results)} schemes"
    )

    for index, scheme in enumerate(graph_results, start=1):

        print(
            f"\n{index}. {scheme.get('title')}"
        )

        print(
            "   Beneficiary: Farmers"
        )

        print(
            "   Benefit Type: Subsidy"
        )


if __name__ == "__main__":
    main()