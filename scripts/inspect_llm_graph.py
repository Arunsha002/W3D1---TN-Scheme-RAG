import pickle
from collections import Counter

import networkx as nx


GRAPH_FILE = "data/raw/llm_knowledge_graph.gpickle"


def load_graph():
    with open(GRAPH_FILE, "rb") as file:
        graph = pickle.load(file)

    return graph


def show_node_types(graph):
    print("\n" + "=" * 70)
    print("NODE TYPES")
    print("=" * 70)

    node_types = Counter()

    for _, data in graph.nodes(data=True):
        node_type = data.get("node_type", "Unknown")
        node_types[node_type] += 1

    for node_type, count in node_types.most_common():
        print(f"{node_type:<20} : {count}")


def show_relationship_types(graph):
    print("\n" + "=" * 70)
    print("RELATIONSHIP TYPES")
    print("=" * 70)

    relationship_types = Counter()

    for _, _, data in graph.edges(data=True):
        relationship = data.get("relationship", "Unknown")
        relationship_types[relationship] += 1

    for relationship, count in relationship_types.most_common():
        print(f"{relationship:<20} : {count}")


def show_sample_schemes(graph, limit=5):
    print("\n" + "=" * 70)
    print(f"SAMPLE SCHEMES (FIRST {limit})")
    print("=" * 70)

    schemes = []

    for node, data in graph.nodes(data=True):
        if data.get("node_type") == "Scheme":
            schemes.append((node, data))

    for node, data in schemes[:limit]:

        print("\n" + "-" * 70)
        print("Scheme:", data.get("name"))

        print("\nRelationships:")

        has_relationship = False

        for source, target, edge_data in graph.out_edges(
            node,
            data=True
        ):
            relationship = edge_data.get("relationship", "Unknown")

            target_data = graph.nodes[target]

            target_name = target_data.get("name", target)
            target_type = target_data.get("node_type", "Unknown")

            print(
                f"  --[{relationship}]--> "
                f"{target_name} ({target_type})"
            )

            has_relationship = True

        if not has_relationship:
            print("  No outgoing relationships found.")


def main():

    print("=" * 70)
    print("LLM KNOWLEDGE GRAPH INSPECTION")
    print("=" * 70)

    print("\nLoading graph...")

    graph = load_graph()

    print("Graph loaded successfully.")

    print("\nTotal nodes:", graph.number_of_nodes())
    print("Total edges:", graph.number_of_edges())

    show_node_types(graph)

    show_relationship_types(graph)

    show_sample_schemes(graph, limit=5)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()