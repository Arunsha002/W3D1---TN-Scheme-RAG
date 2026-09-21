import json
import pickle

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


VECTORSTORE_DIR = "vectorstore"
GRAPH_FILE = "data/raw/llm_knowledge_graph.gpickle"
SCHEMES_FILE = "data/raw/schemes.json"


# --------------------------------------------------
# Query Understanding Model
# --------------------------------------------------

class QueryFilters(BaseModel):

    beneficiary: str | None = Field(
        default=None
    )

    benefit_type: str | None = Field(
        default=None
    )


# --------------------------------------------------
# Query Understanding
# --------------------------------------------------

def understand_question(question):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    structured_llm = llm.with_structured_output(
        QueryFilters
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query understanding component for a
Tamil Nadu Government scheme search system.

Extract the beneficiary and benefit type from
the user's question.

Use only concepts that are mentioned or clearly
implied by the question.

Examples:

Question:
Which schemes provide subsidies to farmers?

Output:
beneficiary = Farmers
benefit_type = Subsidy

Question:
What schemes are available for farmers?

Output:
beneficiary = Farmers
benefit_type = None

Question:
Which schemes provide grants?

Output:
beneficiary = None
benefit_type = Grants

Question:
Which schemes are available for students?

Output:
beneficiary = Students
benefit_type = None

Do not invent values.
If a value cannot be determined, return None.
"""
            ),
            (
                "human",
                """
User question:

{question}
"""
            )
        ]
    )

    messages = prompt.format_messages(
        question=question
    )

    return structured_llm.invoke(
        messages
    )


# --------------------------------------------------
# Load FAISS
# --------------------------------------------------

def load_vectorstore():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


# --------------------------------------------------
# Load Knowledge Graph
# --------------------------------------------------

def load_graph():

    with open(GRAPH_FILE, "rb") as file:
        return pickle.load(file)


# --------------------------------------------------
# Load Original Scheme Data
# --------------------------------------------------

def load_schemes():

    with open(
        SCHEMES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# Convert Scheme to Document
# --------------------------------------------------

def scheme_to_document(scheme):

    page_content = f"""
Scheme Title: {scheme.get("title")}

Department: {scheme.get("department")}

District: {scheme.get("district")}

Organisation: {scheme.get("organisation")}

Associated Scheme: {scheme.get("associated_scheme")}

Sponsored By: {scheme.get("sponsored_by")}

Funding Pattern: {scheme.get("funding_pattern")}

Beneficiaries: {scheme.get("beneficiaries")}

Benefit Type: {scheme.get("benefit_type")}

Eligibility: {scheme.get("eligibility")}

How To Avail: {scheme.get("how_to_avail")}

Validity: {scheme.get("validity")}

Introduced On: {scheme.get("introduced_on")}

Description: {scheme.get("description")}

Scheme Type: {scheme.get("scheme_type")}

Uploaded File: {scheme.get("uploaded_file")}
""".strip()

    metadata = {
        "title": scheme.get("title"),
        "department": scheme.get("department"),
        "beneficiaries": scheme.get("beneficiaries"),
        "benefit_type": scheme.get("benefit_type"),
        "url": scheme.get("url")
    }

    return Document(
        page_content=page_content,
        metadata=metadata
    )


# --------------------------------------------------
# Create Scheme Lookup
# --------------------------------------------------

def create_scheme_lookup(schemes):

    lookup = {}

    for scheme in schemes:

        url = scheme.get("url")

        if url:
            lookup[url] = scheme_to_document(
                scheme
            )

    return lookup


# --------------------------------------------------
# Vector Search
# --------------------------------------------------

def vector_search(
    vectorstore,
    question,
    k=5
):

    return vectorstore.similarity_search(
        question,
        k=k
    )


# --------------------------------------------------
# Graph Search
# --------------------------------------------------

def graph_search(
    graph,
    beneficiary=None,
    benefit_type=None
):

    beneficiary_matches = set()
    benefit_matches = set()

    # ----------------------------------------------
    # Beneficiary
    # ----------------------------------------------

    if beneficiary:

        beneficiary_node = (
            f"Beneficiary:{beneficiary}"
        )

        if graph.has_node(
            beneficiary_node
        ):

            for (
                scheme_node,
                _,
                edge_data
            ) in graph.in_edges(
                beneficiary_node,
                data=True
            ):

                if edge_data.get(
                    "relationship"
                ) == "BENEFITS":

                    beneficiary_matches.add(
                        scheme_node
                    )

    # ----------------------------------------------
    # Benefit Type
    # ----------------------------------------------

    if benefit_type:

        benefit_node = (
            f"Benefit:{benefit_type}"
        )

        if graph.has_node(
            benefit_node
        ):

            for (
                scheme_node,
                _,
                edge_data
            ) in graph.in_edges(
                benefit_node,
                data=True
            ):

                if edge_data.get(
                    "relationship"
                ) in [
                    "HAS_BENEFIT_TYPE",
                    "PROVIDES_BENEFIT"
                ]:

                    benefit_matches.add(
                        scheme_node
                    )

    # ----------------------------------------------
    # Combine
    # ----------------------------------------------

    if beneficiary and benefit_type:

        matching_schemes = (
            beneficiary_matches
            & benefit_matches
        )

    elif beneficiary:

        matching_schemes = (
            beneficiary_matches
        )

    elif benefit_type:

        matching_schemes = (
            benefit_matches
        )

    else:

        matching_schemes = set()

    # ----------------------------------------------
    # Convert to dictionaries
    # ----------------------------------------------

    results = []

    for scheme_node in matching_schemes:

        data = graph.nodes[
            scheme_node
        ]

        results.append({
            "title": data.get("name"),
            "url": data.get("url"),
            "department": data.get(
                "department"
            ),
            "graph_node": scheme_node
        })

    return results


# --------------------------------------------------
# Combine Results
# --------------------------------------------------

def combine_results(
    vector_results,
    graph_results,
    scheme_lookup
):

    combined = {}

    # ----------------------------------------------
    # Vector results
    # ----------------------------------------------

    for document in vector_results:

        url = document.metadata.get(
            "url"
        )

        if not url:
            continue

        combined[url] = {
            "title": document.metadata.get(
                "title"
            ),
            "url": url,
            "department": document.metadata.get(
                "department"
            ),
            "content": document.page_content,
            "source": "vector"
        }

    # ----------------------------------------------
    # Graph results
    # ----------------------------------------------

    for scheme in graph_results:

        url = scheme.get("url")

        if not url:
            continue

        if url in combined:

            combined[url]["source"] = (
                "vector + graph"
            )

        else:

            original_document = (
                scheme_lookup.get(url)
            )

            if original_document:

                combined[url] = {
                    "title":
                        original_document.metadata.get(
                            "title"
                        ),

                    "url":
                        url,

                    "department":
                        original_document.metadata.get(
                            "department"
                        ),

                    "content":
                        original_document.page_content,

                    "source":
                        "graph"
                }

    return list(
        combined.values()
    )


# --------------------------------------------------
# Build Context
# --------------------------------------------------

def build_context(results):

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"""
SCHEME {index}

Title:
{result.get("title")}

Department:
{result.get("department")}

Source:
{result.get("source")}

URL:
{result.get("url")}

Details:
{result.get("content")}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(
    question,
    context
):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Tamil Nadu Government scheme assistant.

Answer the user's question using ONLY the
provided scheme context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context,
   say that the information was not found.
3. Give scheme names clearly.
4. Include relevant details such as benefits,
   eligibility, department or locations when
   available.
5. Treat the retrieved information as source data.
6. Do not claim that a scheme provides a benefit
   unless the context supports it.
"""
            ),
            (
                "human",
                """
User Question:
{question}

Retrieved Scheme Context:
{context}

Answer the question based only on the retrieved
context.
"""
            )
        ]
    )

    messages = prompt.format_messages(
        question=question,
        context=context
    )

    response = llm.invoke(
        messages
    )

    return response.content


# --------------------------------------------------
# Complete Hybrid RAG
# --------------------------------------------------

def hybrid_rag(question):

    print("\n1. Understanding question...")

    filters = understand_question(
        question
    )

    print(
        "   Beneficiary:",
        filters.beneficiary
    )

    print(
        "   Benefit Type:",
        filters.benefit_type
    )

    print("\n2. Loading vector database...")

    vectorstore = load_vectorstore()

    print("3. Loading knowledge graph...")

    graph = load_graph()

    print("4. Loading original scheme data...")

    schemes = load_schemes()

    scheme_lookup = create_scheme_lookup(
        schemes
    )

    print("\n5. Running vector retrieval...")

    vector_results = vector_search(
        vectorstore,
        question,
        k=5
    )

    print("6. Running graph retrieval...")

    graph_results = graph_search(
        graph,
        beneficiary=filters.beneficiary,
        benefit_type=filters.benefit_type
    )

    print("\nVector results:", len(
        vector_results
    ))

    print(
        "Graph results:",
        len(graph_results)
    )

    print("\n7. Combining results...")

    combined_results = combine_results(
        vector_results,
        graph_results,
        scheme_lookup
    )

    print(
        "Unique combined results:",
        len(combined_results)
    )

    print("\n8. Building context...")

    context = build_context(
        combined_results
    )

    print("\n9. Generating final answer...")

    answer = generate_answer(
        question,
        context
    )

    return answer


# --------------------------------------------------
# Test
# --------------------------------------------------

def main():

    load_dotenv()

    print("=" * 70)
    print("COMPLETE HYBRID RAG")
    print("=" * 70)

    question = (
        "Which schemes provide subsidies to farmers?"
    )

    print("\nUser question:")
    print(question)

    answer = hybrid_rag(
        question
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print("\n" + answer)

    print("\n" + "=" * 70)
    print("HYBRID RAG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()