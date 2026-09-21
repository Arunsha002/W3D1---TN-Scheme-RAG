import json
import pickle

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


SCHEMES_FILE = "data/raw/schemes.json"
VECTORSTORE_DIR = "vectorstore"
GRAPH_FILE = "data/raw/llm_knowledge_graph.gpickle"


def load_vectorstore():
    """Load the existing FAISS vector database."""

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
    """Load the LLM-generated Knowledge Graph."""

    with open(GRAPH_FILE, "rb") as file:
        graph = pickle.load(file)

    return graph


def vector_search(vectorstore, question, k=5):
    """Retrieve schemes using semantic vector search."""

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results


def graph_search(graph, beneficiary=None, benefit_type=None):
    """
    Retrieve schemes using Knowledge Graph relationships.

    Currently supports:
    - beneficiary
    - benefit_type
    """

    matching_schemes = set()

    # --------------------------------------------------
    # Find schemes matching beneficiary
    # --------------------------------------------------

    if beneficiary:

        beneficiary_node = f"Beneficiary:{beneficiary}"

        if graph.has_node(beneficiary_node):

            for scheme_node, _, edge_data in graph.in_edges(
                beneficiary_node,
                data=True
            ):

                if edge_data.get("relationship") == "BENEFITS":

                    if graph.nodes[scheme_node].get("node_type") == "Scheme":
                        matching_schemes.add(scheme_node)

    # --------------------------------------------------
    # Find schemes matching benefit type
    # --------------------------------------------------

    if benefit_type:

        benefit_node = f"Benefit:{benefit_type}"

        benefit_matches = set()

        if graph.has_node(benefit_node):

            for scheme_node, _, edge_data in graph.in_edges(
                benefit_node,
                data=True
            ):

                relationship = edge_data.get("relationship")

                if relationship in [
                    "HAS_BENEFIT_TYPE",
                    "PROVIDES_BENEFIT"
                ]:

                    if graph.nodes[scheme_node].get("node_type") == "Scheme":
                        benefit_matches.add(scheme_node)

        # --------------------------------------------------
        # If beneficiary + benefit type are provided,
        # return schemes matching BOTH.
        # --------------------------------------------------

        if beneficiary:
            matching_schemes = matching_schemes.intersection(
                benefit_matches
            )
        else:
            matching_schemes = benefit_matches

    # --------------------------------------------------
    # Convert graph nodes into scheme information
    # --------------------------------------------------

    results = []

    for scheme_node in matching_schemes:

        data = graph.nodes[scheme_node]

        results.append({
            "title": data.get("name"),
            "url": data.get("url"),
            "department": data.get("department"),
            "beneficiaries": data.get("beneficiaries"),
            "benefit_type": data.get("benefit_type"),
            "graph_node": scheme_node
        })

    return results


def hybrid_search(
    question,
    beneficiary=None,
    benefit_type=None,
    vector_k=5
):
    """
    Perform both vector and graph retrieval.
    """

    print("\nLoading vector database...")
    vectorstore = load_vectorstore()

    print("Loading knowledge graph...")
    graph = load_graph()

    # --------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------

    print("\nRunning vector search...")

    vector_results = vector_search(
        vectorstore,
        question,
        k=vector_k
    )

    # --------------------------------------------------
    # Graph retrieval
    # --------------------------------------------------

    print("Running graph search...")

    graph_results = graph_search(
        graph,
        beneficiary=beneficiary,
        benefit_type=benefit_type
    )

    return vector_results, graph_results


def main():

    load_dotenv()

    print("=" * 70)
    print("HYBRID RETRIEVER TEST")
    print("=" * 70)

    question = "Which schemes provide subsidies to farmers?"

    print("\nQuestion:")
    print(question)

    # For this test we explicitly provide the graph filters.
    beneficiary = "Farmers"
    benefit_type = "Subsidy"

    vector_results, graph_results = hybrid_search(
        question=question,
        beneficiary=beneficiary,
        benefit_type=benefit_type,
        vector_k=5
    )

    # --------------------------------------------------
    # Display vector results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("VECTOR SEARCH RESULTS")
    print("=" * 70)

    for index, document in enumerate(
        vector_results,
        start=1
    ):

        print(f"\n{index}. {document.metadata.get('title')}")
        print("   Department:", document.metadata.get("department"))

    # --------------------------------------------------
    # Display graph results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAPH SEARCH RESULTS")
    print("=" * 70)

    for index, scheme in enumerate(
        graph_results,
        start=1
    ):

        print(f"\n{index}. {scheme['title']}")
        print("   Department:", scheme["department"])
        print("   Beneficiaries:", scheme["beneficiaries"])
        print("   Benefit Type:", scheme["benefit_type"])

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL SUMMARY")
    print("=" * 70)

    print("\nVector results:", len(vector_results))
    print("Graph results:", len(graph_results))

    print("\nHybrid retrieval completed successfully.")


if __name__ == "__main__":
    main()