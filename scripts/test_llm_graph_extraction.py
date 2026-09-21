import json
import os
import pickle
import time

import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


INPUT_FILE = "data/raw/schemes.json"
OUTPUT_FILE = "data/raw/llm_knowledge_graph.gpickle"


# ============================================================
# Pydantic models
# ============================================================

class Entity(BaseModel):
    name: str
    type: str


class Relationship(BaseModel):
    source: str
    relationship: str
    target: str


class KnowledgeGraphData(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


# ============================================================
# Load all schemes
# ============================================================

def load_schemes():

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Extract graph information from one scheme
# ============================================================

def extract_graph_data(
    scheme,
    structured_llm,
    prompt
):

    messages = prompt.format_messages(
        title=scheme.get("title"),
        department=scheme.get("department"),
        district=scheme.get("district"),
        organisation=scheme.get("organisation"),
        associated_scheme=scheme.get("associated_scheme"),
        sponsored_by=scheme.get("sponsored_by"),
        funding_pattern=scheme.get("funding_pattern"),
        beneficiaries=scheme.get("beneficiaries"),
        benefit_type=scheme.get("benefit_type"),
        eligibility=scheme.get("eligibility"),
        how_to_avail=scheme.get("how_to_avail"),
        validity=scheme.get("validity"),
        introduced_on=scheme.get("introduced_on"),
        description=scheme.get("description"),
        scheme_type=scheme.get("scheme_type")
    )

    return structured_llm.invoke(messages)


# ============================================================
# Add extracted data to NetworkX graph
# ============================================================

def add_to_graph(
    graph,
    graph_data,
    scheme
):

    # --------------------------------------------------------
    # Add entities
    # --------------------------------------------------------

    for entity in graph_data.entities:

        node_id = f"{entity.type}:{entity.name}"

        graph.add_node(
            node_id,
            name=entity.name,
            node_type=entity.type
        )

    # --------------------------------------------------------
    # Add relationships
    # --------------------------------------------------------

    for relationship in graph_data.relationships:

        source_node = None
        target_node = None

        # Find source entity
        for entity in graph_data.entities:

            if entity.name == relationship.source:

                source_node = (
                    f"{entity.type}:{entity.name}"
                )

            if entity.name == relationship.target:

                target_node = (
                    f"{entity.type}:{entity.name}"
                )

        # Skip relationships where the LLM
        # referenced an entity that it did not
        # include in the entities list.
        if not source_node or not target_node:
            continue

        graph.add_edge(
            source_node,
            target_node,
            relationship=relationship.relationship
        )

    # --------------------------------------------------------
    # Make sure the scheme URL is available
    # --------------------------------------------------------

    scheme_title = scheme.get("title")

    scheme_node = f"Scheme:{scheme_title}"

    if graph.has_node(scheme_node):

        graph.nodes[scheme_node]["url"] = scheme.get("url")

        graph.nodes[scheme_node]["department"] = (
            scheme.get("department")
        )

        graph.nodes[scheme_node]["beneficiaries"] = (
            scheme.get("beneficiaries")
        )

        graph.nodes[scheme_node]["benefit_type"] = (
            scheme.get("benefit_type")
        )


# ============================================================
# Main
# ============================================================

def main():

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY was not found in .env"
        )

    print("=" * 70)
    print("LLM KNOWLEDGE GRAPH BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load schemes
    # --------------------------------------------------------

    schemes = load_schemes()

    print(
        "\nTotal schemes:",
        len(schemes)
    )

    # --------------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    structured_llm = llm.with_structured_output(
        KnowledgeGraphData
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You extract structured knowledge from
Tamil Nadu Government scheme information.

Read the scheme information carefully.

Extract ONLY information explicitly supported
by the supplied scheme.

Do NOT invent information.

Return entities and relationships.

Allowed entity types:

- Scheme
- Department
- Beneficiary
- Benefit
- Crop
- Location
- Organization
- Sponsor

Allowed relationship types:

- BELONGS_TO
- BENEFITS
- HAS_BENEFIT_TYPE
- SPONSORED_BY
- TARGETS_CROP
- TARGETS_LOCATION
- PROVIDES_BENEFIT
- RELATED_TO

Use the exact names from the source whenever possible.

Do not create generic entities such as:
"Department", "Scheme", "Government", etc.
unless that exact thing is actually a meaningful entity
in the source.

Only create a relationship when the source supports it.
"""
            ),
            (
                "human",
                """
Scheme Information
==============================

Title:
{title}

Department:
{department}

District:
{district}

Organisation:
{organisation}

Associated Scheme:
{associated_scheme}

Sponsored By:
{sponsored_by}

Funding Pattern:
{funding_pattern}

Beneficiaries:
{beneficiaries}

Benefit Type:
{benefit_type}

Eligibility:
{eligibility}

How To Avail:
{how_to_avail}

Validity:
{validity}

Introduced On:
{introduced_on}

Description:
{description}

Scheme Type:
{scheme_type}

==============================

Extract the knowledge graph information.
"""
            )
        ]
    )

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    graph = nx.MultiDiGraph()

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process each scheme
    # --------------------------------------------------------

    for index, scheme in enumerate(
        schemes,
        start=1
    ):

        title = scheme.get(
            "title",
            "Unknown Scheme"
        )

        print(
            f"\n[{index}/{len(schemes)}] "
            f"{title}"
        )

        try:

            graph_data = extract_graph_data(
                scheme,
                structured_llm,
                prompt
            )

            add_to_graph(
                graph,
                graph_data,
                scheme
            )

            successful += 1

            print(
                "    ✓ Extraction successful"
            )

            print(
                f"    Entities: "
                f"{len(graph_data.entities)}"
            )

            print(
                f"    Relationships: "
                f"{len(graph_data.relationships)}"
            )

        except Exception as error:

            failed += 1

            print(
                "    ✗ Extraction failed"
            )

            print(
                f"    Error: {error}"
            )

        # Small pause between API calls
        time.sleep(0.2)

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "wb"
    ) as file:

        pickle.dump(
            graph,
            file
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("KNOWLEDGE GRAPH BUILD COMPLETE")
    print("=" * 70)

    print(
        "\nSuccessful schemes:",
        successful
    )

    print(
        "Failed schemes:",
        failed
    )

    print(
        "Graph nodes:",
        graph.number_of_nodes()
    )

    print(
        "Graph edges:",
        graph.number_of_edges()
    )

    print(
        "\nGraph saved to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()