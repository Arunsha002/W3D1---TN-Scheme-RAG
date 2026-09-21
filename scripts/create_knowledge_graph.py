import json
import networkx as nx

INPUT_FILE = "data/raw/schemes.json"
OUTPUT_FILE = "data/raw/knowledge_graph.gpickle"


def load_schemes():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def create_graph(schemes):
    graph = nx.MultiDiGraph()

    for scheme in schemes:
        title = scheme.get("title")

        if not title:
            continue

        # --------------------------------------------------
        # 1. Create Scheme node
        # --------------------------------------------------

        scheme_node = f"scheme:{title}"

        graph.add_node(
            scheme_node,
            node_type="scheme",
            title=title,
            url=scheme.get("url"),
            description=scheme.get("description"),
            funding_pattern=scheme.get("funding_pattern"),
            how_to_avail=scheme.get("how_to_avail"),
        )

        # --------------------------------------------------
        # 2. Department relationship
        # --------------------------------------------------

        department = scheme.get("department")

        if department:
            department_node = f"department:{department}"

            graph.add_node(
                department_node,
                node_type="department",
                name=department
            )

            graph.add_edge(
                scheme_node,
                department_node,
                relationship="BELONGS_TO"
            )

        # --------------------------------------------------
        # 3. Beneficiary relationship
        # --------------------------------------------------

        beneficiary = scheme.get("beneficiaries")

        if beneficiary:
            beneficiary_node = f"beneficiary:{beneficiary}"

            graph.add_node(
                beneficiary_node,
                node_type="beneficiary",
                name=beneficiary
            )

            graph.add_edge(
                scheme_node,
                beneficiary_node,
                relationship="BENEFITS"
            )

        # --------------------------------------------------
        # 4. Benefit Type relationship
        # --------------------------------------------------

        benefit_type = scheme.get("benefit_type")

        if benefit_type:
            benefit_node = f"benefit:{benefit_type}"

            graph.add_node(
                benefit_node,
                node_type="benefit_type",
                name=benefit_type
            )

            graph.add_edge(
                scheme_node,
                benefit_node,
                relationship="HAS_BENEFIT_TYPE"
            )

        # --------------------------------------------------
        # 5. Sponsored By relationship
        # --------------------------------------------------

        sponsored_by = scheme.get("sponsored_by")

        if sponsored_by:
            sponsor_node = f"sponsor:{sponsored_by}"

            graph.add_node(
                sponsor_node,
                node_type="sponsor",
                name=sponsored_by
            )

            graph.add_edge(
                scheme_node,
                sponsor_node,
                relationship="SPONSORED_BY"
            )

    return graph


def main():

    print("Loading schemes...")

    schemes = load_schemes()

    print("Total schemes:", len(schemes))

    print("\nCreating Knowledge Graph...")

    graph = create_graph(schemes)

    print("\nKnowledge Graph created successfully.")

    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    # Display a few nodes
    print("\nSample nodes:")

    for node, data in list(graph.nodes(data=True))[:10]:
        print(node, "->", data)

    # Display a few relationships
    print("\nSample relationships:")

    count = 0

    for source, target, data in graph.edges(data=True):

        print(
            source,
            "--",
            data["relationship"],
            "-->",
            target
        )

        count += 1

        if count >= 10:
            break

    # Save graph
    with open(OUTPUT_FILE, "wb") as file:
        import pickle
        pickle.dump(graph, file)

    print("\nKnowledge Graph saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()