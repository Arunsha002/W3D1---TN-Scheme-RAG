import pickle
import networkx as nx

GRAPH_FILE = "data/raw/knowledge_graph.gpickle"


def load_graph():
    with open(GRAPH_FILE, "rb") as file:
        return pickle.load(file)


def find_subsidy_schemes(graph):
    results = []

    # Find all scheme nodes
    for scheme_node, scheme_data in graph.nodes(data=True):

        if scheme_data.get("node_type") != "scheme":
            continue

        # Check the relationships of this scheme
        for _, target_node, edge_data in graph.out_edges(
            scheme_node,
            data=True
        ):

            relationship = edge_data.get("relationship")

            # We are looking for:
            # Scheme -> BENEFITS -> Farmers
            if relationship == "BENEFITS":

                beneficiary_data = graph.nodes[target_node]

                if beneficiary_data.get("name") != "Farmers":
                    continue

                # Now check whether the same scheme
                # has a HAS_BENEFIT_TYPE -> Subsidy relationship
                for _, benefit_node, benefit_edge_data in graph.out_edges(
                    scheme_node,
                    data=True
                ):

                    if (
                        benefit_edge_data.get("relationship")
                        == "HAS_BENEFIT_TYPE"
                    ):

                        benefit_data = graph.nodes[benefit_node]

                        if benefit_data.get("name") == "Subsidy":
                            results.append(scheme_data)

    return results


def main():

    print("Loading Knowledge Graph...")

    graph = load_graph()

    print("Nodes:", graph.number_of_nodes())
    print("Edges:", graph.number_of_edges())

    print("\nSearching for schemes where:")
    print("Beneficiary = Farmers")
    print("Benefit Type = Subsidy")

    results = find_subsidy_schemes(graph)

    print("\nMatching schemes:", len(results))

    print("\nResults:")
    print("-" * 60)

    for index, scheme in enumerate(results, start=1):

        print(f"\n{index}. {scheme['title']}")

        print("   URL:", scheme.get("url"))

        print("   Description:")
        print("   ", scheme.get("description"))

        print("   Funding Pattern:")
        print("   ", scheme.get("funding_pattern"))


if __name__ == "__main__":
    main()