import pickle

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


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

            relationships.append({
                "relationship": edge_data.get("relationship"),
                "target": target_data.get("name")
            })

        break

    return relationships


def build_hybrid_results(
    vector_documents,
    graph,
    graph_schemes
):

    schemes = {}

    # ------------------------------------------------
    # 1. Add vector search results
    # ------------------------------------------------

    for rank, document in enumerate(
        vector_documents,
        start=1
    ):

        url = document.metadata.get("url")

        if not url:
            continue

        schemes[url] = {
            "title": document.metadata.get("title"),
            "url": url,
            "content": document.page_content,
            "graph_relationships": [],
            "vector_rank": rank,
            "graph_match": False
        }

    # ------------------------------------------------
    # 2. Add graph results
    # ------------------------------------------------

    for scheme in graph_schemes:

        url = scheme.get("url")

        if not url:
            continue

        relationships = get_graph_relationships(
            graph,
            url
        )

        if url in schemes:

            # Scheme was found by BOTH methods
            schemes[url]["graph_relationships"] = relationships
            schemes[url]["graph_match"] = True

        else:

            # Scheme was found only by graph
            schemes[url] = {
                "title": scheme.get("title"),
                "url": url,
                "content": f"""
Scheme Title:
{scheme.get("title")}

Description:
{scheme.get("description")}

Funding Pattern:
{scheme.get("funding_pattern")}

How To Avail:
{scheme.get("how_to_avail")}
""".strip(),
                "graph_relationships": relationships,
                "vector_rank": None,
                "graph_match": True
            }

    return schemes


def calculate_score(scheme):

    score = 0

    # ---------------------------------------------
    # Graph match is strong evidence
    # ---------------------------------------------

    if scheme["graph_match"]:
        score += 100

    # ---------------------------------------------
    # Vector ranking
    #
    # Rank 1 = highest score
    # Rank 5 = lower score
    # ---------------------------------------------

    vector_rank = scheme["vector_rank"]

    if vector_rank is not None:

        vector_score = 50 - (
            (vector_rank - 1) * 10
        )

        score += vector_score

    return score


def format_context(schemes):

    context_parts = []

    for index, scheme in enumerate(
        schemes,
        start=1
    ):

        context = f"""
SCHEME {index}
==============================

Title:
{scheme["title"]}

URL:
{scheme["url"]}

Relevance Score:
{scheme["score"]}

Scheme Information:
{scheme["content"]}

Knowledge Graph Relationships:
"""

        relationships = scheme.get(
            "graph_relationships",
            []
        )

        if relationships:

            for relationship in relationships:

                context += (
                    f"- "
                    f"{relationship['relationship']}"
                    f" -> "
                    f"{relationship['target']}\n"
                )

        else:

            context += "- None\n"

        context_parts.append(context)

    return "\n\n".join(context_parts)


def main():

    load_dotenv()

    question = "Which schemes provide subsidies to farmers?"

    # ==================================================
    # VECTOR RETRIEVAL
    # ==================================================

    print("=" * 70)
    print("VECTOR RETRIEVAL")
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
    print("GRAPH RETRIEVAL")
    print("=" * 70)

    graph = load_graph()

    graph_schemes = graph_search(graph)

    print(
        "Graph schemes:",
        len(graph_schemes)
    )

    # ==================================================
    # HYBRID RETRIEVAL
    # ==================================================

    print("\n")
    print("=" * 70)
    print("BUILDING HYBRID RESULTS")
    print("=" * 70)

    schemes = build_hybrid_results(
        vector_documents,
        graph,
        graph_schemes
    )

    print(
        "Unique schemes:",
        len(schemes)
    )

    # ==================================================
    # RANKING
    # ==================================================

    print("\n")
    print("=" * 70)
    print("RANKING RESULTS")
    print("=" * 70)

    ranked_schemes = []

    for scheme in schemes.values():

        score = calculate_score(scheme)

        scheme["score"] = score

        ranked_schemes.append(scheme)

    ranked_schemes.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # --------------------------------------------------
    # Display ranking
    # --------------------------------------------------

    for index, scheme in enumerate(
        ranked_schemes,
        start=1
    ):

        print(
            f"{index}. "
            f"{scheme['title']} "
            f"→ Score: {scheme['score']} "
            f"| Graph: {scheme['graph_match']} "
            f"| Vector Rank: {scheme['vector_rank']}"
        )

    # ==================================================
    # SELECT TOP RESULTS
    # ==================================================

    top_schemes = ranked_schemes[:10]

    print("\n")
    print("=" * 70)
    print("TOP RESULTS SENT TO GPT")
    print("=" * 70)

    for index, scheme in enumerate(
        top_schemes,
        start=1
    ):

        print(
            f"{index}. {scheme['title']}"
        )

    # ==================================================
    # BUILD CONTEXT
    # ==================================================

    context = format_context(
        top_schemes
    )

    # ==================================================
    # GPT-4o-mini
    # ==================================================

    print("\n")
    print("=" * 70)
    print("CALLING GPT-4o-mini")
    print("=" * 70)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful assistant for Tamil Nadu
Government schemes.

Answer the user's question using ONLY the
information provided in the hybrid context.

The context contains:

1. Vector search results.
2. Knowledge Graph relationships.

IMPORTANT:

A Knowledge Graph relationship is stronger
evidence than a general semantic match.

Do not invent scheme names, eligibility rules,
benefits, amounts, or application procedures.

If the information is not available in the
context, say that it could not be found.

Keep the answer clear and concise.
"""
            ),
            (
                "human",
                """
Hybrid Ranked Context
==============================

{context}

==============================

User Question:

{question}

Answer:
"""
            )
        ]
    )

    messages = prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(messages)

    # ==================================================
    # FINAL ANSWER
    # ==================================================

    print("\n")
    print("=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print("\n")

    print(response.content)


if __name__ == "__main__":
    main()