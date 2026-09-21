import pickle


GRAPH_FILE = "data/raw/knowledge_graph.gpickle"


def load_graph():

    with open(GRAPH_FILE, "rb") as file:
        return pickle.load(file)


def get_graph_values(graph):

    beneficiaries = set()
    benefit_types = set()

    for node, data in graph.nodes(data=True):

        if data.get("node_type") == "beneficiary":
            beneficiaries.add(data.get("name"))

        if data.get("node_type") == "benefit_type":
            benefit_types.add(data.get("name"))

    return beneficiaries, benefit_types


def understand_question(question, graph):

    question_lower = question.lower()

    beneficiaries, benefit_types = get_graph_values(graph)

    filters = {}

    # ---------------------------------------------
    # Detect beneficiary from graph data
    # ---------------------------------------------

    for beneficiary in beneficiaries:

        if not beneficiary:
            continue

        if beneficiary.lower() in question_lower:

            filters["beneficiary"] = beneficiary
            break

    # ---------------------------------------------
    # Detect benefit type from graph data
    # ---------------------------------------------

    for benefit_type in benefit_types:

        if not benefit_type:
            continue

        if benefit_type.lower() in question_lower:

            filters["benefit_type"] = benefit_type
            break

    return filters


def graph_search(graph, filters):

    results = []

    for scheme_node, scheme_data in graph.nodes(data=True):

        if scheme_data.get("node_type") != "scheme":
            continue

        beneficiary_match = True
        benefit_type_match = True

        # -----------------------------------------
        # Check beneficiary
        # -----------------------------------------

        if "beneficiary" in filters:

            beneficiary_match = False

            for _, target_node, edge_data in graph.out_edges(
                scheme_node,
                data=True
            ):

                target_data = graph.nodes[target_node]

                if (
                    edge_data.get("relationship") == "BENEFITS"
                    and target_data.get("name")
                    == filters["beneficiary"]
                ):

                    beneficiary_match = True
                    break

        # -----------------------------------------
        # Check benefit type
        # -----------------------------------------

        if "benefit_type" in filters:

            benefit_type_match = False

            for _, target_node, edge_data in graph.out_edges(
                scheme_node,
                data=True
            ):

                target_data = graph.nodes[target_node]

                if (
                    edge_data.get("relationship")
                    == "HAS_BENEFIT_TYPE"
                    and target_data.get("name")
                    == filters["benefit_type"]
                ):

                    benefit_type_match = True
                    break

        # -----------------------------------------
        # Add matching scheme
        # -----------------------------------------

        if beneficiary_match and benefit_type_match:

            results.append(scheme_data)

    return results


def main():

    graph = load_graph()

    print("=" * 70)
    print("QUERY-AWARE KNOWLEDGE GRAPH")
    print("=" * 70)

    # ---------------------------------------------
    # Test question
    # ---------------------------------------------

    question = "Which schemes provide subsidies to farmers?"

    print("\nUser Question:")
    print(question)

    # ---------------------------------------------
    # Understand question
    # ---------------------------------------------

    filters = understand_question(
        question,
        graph
    )

    print("\nDetected Filters:")
    print(filters)

    # ---------------------------------------------
    # Search graph
    # ---------------------------------------------

    results = graph_search(
        graph,
        filters
    )

    print("\nGraph Results:")
    print(
        "Matching schemes:",
        len(results)
    )

    # ---------------------------------------------
    # Display results
    # ---------------------------------------------

    for index, scheme in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{scheme.get('title')}"
        )


if __name__ == "__main__":
    main()