import json
import os
import pickle
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ENV_FILE = PROJECT_ROOT / ".env"

SCHEMES_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "schemes.json"
)

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "vectorstore"
)

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "llm_knowledge_graph.gpickle"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(ENV_FILE)


# ============================================================
# QUERY FILTERS
# ============================================================

class QueryFilters(BaseModel):

    beneficiary: str | None = None

    benefit_type: str | None = None

    department: str | None = None

    location: str | None = None

    sponsor: str | None = None

    crop: str | None = None

    scheme_type: str | None = None


# ============================================================
# RAG SERVICE
# ============================================================

class RAGService:

    def __init__(self):

        print("=" * 70)
        print("LOADING TN SCHEME RAG SERVICE")
        print("=" * 70)

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        if not os.getenv("OPENAI_API_KEY"):

            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please check your .env file."
            )

        # ----------------------------------------------------
        # SCHEME DATA
        # ----------------------------------------------------

        print("Loading scheme data...")

        with open(
            SCHEMES_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.schemes = json.load(file)

        print(
            f"Loaded {len(self.schemes)} schemes."
        )

        # ----------------------------------------------------
        # SCHEME LOOKUP
        # ----------------------------------------------------

        self.scheme_lookup = {}

        for scheme in self.schemes:

            url = scheme.get("url")

            if url:

                self.scheme_lookup[url] = scheme

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        print("Loading OpenAI embeddings...")

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        # ----------------------------------------------------
        # FAISS
        # ----------------------------------------------------

        print("Loading FAISS vector store...")

        self.vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        print(
            "FAISS vector store loaded."
        )

        # ----------------------------------------------------
        # KNOWLEDGE GRAPH
        # ----------------------------------------------------

        print("Loading knowledge graph...")

        with open(
            GRAPH_FILE,
            "rb"
        ) as file:

            self.graph = pickle.load(file)

        print(
            f"Knowledge graph loaded: "
            f"{self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges."
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        print("Loading GPT-4o-mini...")

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        # ----------------------------------------------------
        # STRUCTURED QUERY LLM
        # ----------------------------------------------------

        self.query_llm = self.llm.with_structured_output(
            QueryFilters
        )

        print("=" * 70)
        print("RAG SERVICE READY")
        print("=" * 70)

    # ========================================================
    # NORMALIZE TEXT
    # ========================================================

    @staticmethod
    def normalize_text(value) -> str:

        if value is None:

            return ""

        return " ".join(
            str(value)
            .strip()
            .lower()
            .split()
        )

    # ========================================================
    # VALUE CONTAINS
    # ========================================================

    @staticmethod
    def value_contains(
        source_value,
        search_value
    ) -> bool:

        source = RAGService.normalize_text(
            source_value
        )

        search = RAGService.normalize_text(
            search_value
        )

        if not source or not search:

            return False

        return search in source

    # ========================================================
    # FORMAT CONVERSATION HISTORY
    # ========================================================

    def format_conversation_history(
        self,
        conversation_history
    ):

        if not conversation_history:

            return "No previous conversation."

        recent_history = (
            conversation_history[-6:]
        )

        lines = []

        for message in recent_history:

            role = message.get(
                "role",
                "user"
            )

            content = message.get(
                "content",
                ""
            )

            if not content:

                continue

            if role == "assistant":

                label = "Assistant"

            else:

                label = "User"

            lines.append(
                f"{label}: {content}"
            )

        if not lines:

            return "No previous conversation."

        return "\n".join(lines)

    # ========================================================
    # QUERY UNDERSTANDING
    # ========================================================

    def understand_query(
        self,
        question,
        conversation_history
    ) -> QueryFilters:

        history_text = (
            self.format_conversation_history(
                conversation_history
            )
        )

        prompt = f"""
You are a query understanding system for a
Tamil Nadu Government Scheme database.

Extract structured filters from the user's
CURRENT question.

Available filters:

- beneficiary
- benefit_type
- department
- location
- sponsor
- crop
- scheme_type

IMPORTANT CONVERSATION RULE:

The user may ask follow-up questions.

If the current question clearly refers to
the previous conversation, inherit relevant
filters from the previous user question.

For example:

Previous:
Which schemes provide subsidies to farmers?

Current:
What about pulses?

Output should be approximately:

beneficiary = Farmers
benefit_type = Subsidy
crop = Pulses

Another example:

Previous:
Which agriculture schemes are available?

Current:
What about Chennai?

Output:

department = Agriculture - Farmers Welfare Department
location = Chennai

If the current question introduces a new
filter, add it.

If the current question explicitly changes
a previous filter, use the new value.

Do not invent filters when there is no evidence.

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

Question:
Which schemes provide grants?

Output:
benefit_type = Grants

Question:
Which schemes are available for farmers in Chennai?

Output:
beneficiary = Farmers
location = Chennai

Question:
Which schemes support pulses?

Output:
crop = Pulses

Question:
Which schemes are sponsored by the State?

Output:
sponsor = State

Question:
Which agriculture schemes provide subsidies?

Output:
department = Agriculture - Farmers Welfare Department
benefit_type = Subsidy

Previous conversation:

{history_text}

CURRENT USER QUESTION:

{question}
"""

        filters = self.query_llm.invoke(
            prompt
        )

        return filters

    # ========================================================
    # FIND GRAPH NODES
    # ========================================================

    def find_graph_nodes(
        self,
        node_type,
        value
    ):

        target_value = self.normalize_text(
            value
        )

        matching_nodes = []

        for node, data in self.graph.nodes(
            data=True
        ):

            node_text = str(node)

            if ":" in node_text:

                prefix, name = node_text.split(
                    ":",
                    1
                )

                if (
                    prefix.lower()
                    == node_type.lower()
                ):

                    if (
                        self.normalize_text(name)
                        == target_value
                    ):

                        matching_nodes.append(
                            node
                        )

                        continue

            actual_type = data.get(
                "type"
            )

            actual_name = data.get(
                "name"
            )

            if (
                actual_type
                and actual_name
                and str(actual_type).lower()
                == node_type.lower()
            ):

                if (
                    self.normalize_text(actual_name)
                    == target_value
                ):

                    matching_nodes.append(
                        node
                    )

        return matching_nodes

    # ========================================================
    # GET CONNECTED SCHEMES
    # ========================================================

    def get_connected_schemes(
        self,
        entity_nodes,
        allowed_relationships
    ):

        scheme_nodes = set()

        for entity_node in entity_nodes:

            for _, target, data in self.graph.out_edges(
                entity_node,
                data=True
            ):

                relationship = data.get(
                    "relationship"
                )

                if (
                    relationship
                    not in allowed_relationships
                ):

                    continue

                if str(target).startswith(
                    "Scheme:"
                ):

                    scheme_nodes.add(
                        target
                    )

            for source, _, data in self.graph.in_edges(
                entity_node,
                data=True
            ):

                relationship = data.get(
                    "relationship"
                )

                if (
                    relationship
                    not in allowed_relationships
                ):

                    continue

                if str(source).startswith(
                    "Scheme:"
                ):

                    scheme_nodes.add(
                        source
                    )

        return scheme_nodes

    # ========================================================
    # GRAPH SEARCH
    # ========================================================

    def graph_search(
        self,
        filters
    ):

        filter_sets = []

        # ----------------------------------------------------
        # Beneficiary
        # ----------------------------------------------------

        if filters.beneficiary:

            nodes = self.find_graph_nodes(
                "Beneficiary",
                filters.beneficiary
            )

            schemes = self.get_connected_schemes(
                nodes,
                {"BENEFITS"}
            )

            filter_sets.append(
                schemes
            )

        # ----------------------------------------------------
        # Benefit
        # ----------------------------------------------------

        if filters.benefit_type:

            nodes = self.find_graph_nodes(
                "Benefit",
                filters.benefit_type
            )

            schemes = self.get_connected_schemes(
                nodes,
                {
                    "HAS_BENEFIT_TYPE",
                    "PROVIDES_BENEFIT"
                }
            )

            filter_sets.append(
                schemes
            )

        # ----------------------------------------------------
        # Department
        # ----------------------------------------------------

        if filters.department:

            nodes = self.find_graph_nodes(
                "Department",
                filters.department
            )

            schemes = self.get_connected_schemes(
                nodes,
                {"BELONGS_TO"}
            )

            filter_sets.append(
                schemes
            )

        # ----------------------------------------------------
        # Location
        # ----------------------------------------------------

        if filters.location:

            nodes = self.find_graph_nodes(
                "Location",
                filters.location
            )

            schemes = self.get_connected_schemes(
                nodes,
                {"TARGETS_LOCATION"}
            )

            filter_sets.append(
                schemes
            )

        # ----------------------------------------------------
        # Sponsor
        # ----------------------------------------------------

        if filters.sponsor:

            nodes = self.find_graph_nodes(
                "Sponsor",
                filters.sponsor
            )

            schemes = self.get_connected_schemes(
                nodes,
                {"SPONSORED_BY"}
            )

            filter_sets.append(
                schemes
            )

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        if filters.crop:

            nodes = self.find_graph_nodes(
                "Crop",
                filters.crop
            )

            schemes = self.get_connected_schemes(
                nodes,
                {"TARGETS_CROP"}
            )

            filter_sets.append(
                schemes
            )

        # ----------------------------------------------------
        # No graph filters
        # ----------------------------------------------------

        if not filter_sets:

            return []

        # ----------------------------------------------------
        # INTERSECTION
        # ----------------------------------------------------

        matching_schemes = filter_sets[0]

        for scheme_set in filter_sets[1:]:

            matching_schemes = (
                matching_schemes
                & scheme_set
            )

        # ----------------------------------------------------
        # CONVERT TO SOURCE RECORDS
        # ----------------------------------------------------

        results = []

        for scheme_node in matching_schemes:

            scheme_url = self.graph.nodes[
                scheme_node
            ].get(
                "url"
            )

            if not scheme_url:

                continue

            scheme = self.scheme_lookup.get(
                scheme_url
            )

            if not scheme:

                continue

            results.append(
                {
                    "scheme": scheme,
                    "source": "graph"
                }
            )

        return results

    # ========================================================
    # STRUCTURED SEARCH
    # ========================================================

    def structured_search(
        self,
        filters
    ):

        filters_to_apply = [
            (
                "beneficiary",
                "beneficiaries",
                filters.beneficiary
            ),
            (
                "benefit_type",
                "benefit_type",
                filters.benefit_type
            ),
            (
                "department",
                "department",
                filters.department
            ),
            (
                "location",
                "district",
                filters.location
            ),
            (
                "sponsor",
                "sponsored_by",
                filters.sponsor
            ),
            (
                "scheme_type",
                "scheme_type",
                filters.scheme_type
            )
        ]

        active_filters = [
            item
            for item in filters_to_apply
            if item[2]
        ]

        if not active_filters:

            return []

        results = []

        for scheme in self.schemes:

            matches = True

            for _, field, value in active_filters:

                if not self.value_contains(
                    scheme.get(field),
                    value
                ):

                    matches = False

                    break

            if matches:

                results.append(
                    {
                        "scheme": scheme,
                        "source": "structured"
                    }
                )

        return results

    # ========================================================
    # RESULT VALIDATION
    # ========================================================

    def validate_result(
        self,
        scheme,
        filters
    ):

        if filters.beneficiary:

            if not self.value_contains(
                scheme.get("beneficiaries"),
                filters.beneficiary
            ):

                return False

        if filters.benefit_type:

            if not self.value_contains(
                scheme.get("benefit_type"),
                filters.benefit_type
            ):

                return False

        if filters.department:

            if not self.value_contains(
                scheme.get("department"),
                filters.department
            ):

                return False

        if filters.location:

            location_match = (
                self.value_contains(
                    scheme.get("district"),
                    filters.location
                )
                or
                self.value_contains(
                    scheme.get("description"),
                    filters.location
                )
            )

            if not location_match:

                return False

        if filters.sponsor:

            if not self.value_contains(
                scheme.get("sponsored_by"),
                filters.sponsor
            ):

                return False

        if filters.scheme_type:

            if not self.value_contains(
                scheme.get("scheme_type"),
                filters.scheme_type
            ):

                return False

        if filters.crop:

            crop_match = (
                self.value_contains(
                    scheme.get("description"),
                    filters.crop
                )
                or
                self.value_contains(
                    scheme.get("title"),
                    filters.crop
                )
            )

            if not crop_match:

                return False

        return True

    # ========================================================
    # VALIDATE GRAPH RESULTS
    # ========================================================

    def validate_graph_results(
        self,
        graph_results,
        filters
    ):

        validated = []

        for result in graph_results:

            if self.validate_result(
                result["scheme"],
                filters
            ):

                validated.append(
                    result
                )

        return validated

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    def vector_search(
        self,
        question,
        k=5
    ):

        documents = self.vectorstore.similarity_search(
            question,
            k=k
        )

        results = []

        for document in documents:

            metadata = document.metadata

            url = metadata.get(
                "url"
            )

            scheme = self.scheme_lookup.get(
                url
            )

            if scheme:

                results.append(
                    {
                        "scheme": scheme,
                        "source": "vector"
                    }
                )

        return results

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    def combine_results(
        self,
        vector_results,
        graph_results,
        structured_results
    ):

        combined = {}

        # ----------------------------------------------------
        # VECTOR
        # ----------------------------------------------------

        for result in vector_results:

            scheme = result["scheme"]

            url = scheme.get("url")

            if not url:

                continue

            combined[url] = {
                "scheme": scheme,
                "retrieved_by": "vector"
            }

        # ----------------------------------------------------
        # GRAPH
        # ----------------------------------------------------

        for result in graph_results:

            scheme = result["scheme"]

            url = scheme.get("url")

            if not url:

                continue

            if url in combined:

                existing = combined[url][
                    "retrieved_by"
                ]

                if existing == "vector":

                    combined[url][
                        "retrieved_by"
                    ] = "vector + graph"

                elif "graph" not in existing:

                    combined[url][
                        "retrieved_by"
                    ] = (
                        existing
                        + " + graph"
                    )

            else:

                combined[url] = {
                    "scheme": scheme,
                    "retrieved_by": "graph"
                }

        # ----------------------------------------------------
        # STRUCTURED
        # ----------------------------------------------------

        for result in structured_results:

            scheme = result["scheme"]

            url = scheme.get("url")

            if not url:

                continue

            if url in combined:

                existing = combined[url][
                    "retrieved_by"
                ]

                if "structured" not in existing:

                    combined[url][
                        "retrieved_by"
                    ] = (
                        existing
                        + " + structured"
                    )

            else:

                combined[url] = {
                    "scheme": scheme,
                    "retrieved_by": "structured"
                }

        return list(
            combined.values()
        )

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        results
    ):

        context_parts = []

        for index, result in enumerate(
            results,
            start=1
        ):

            scheme = result["scheme"]

            context_parts.append(
                f"""
SCHEME {index}

Title:
{scheme.get("title")}

Department:
{scheme.get("department")}

District:
{scheme.get("district")}

Organisation:
{scheme.get("organisation")}

Associated Scheme:
{scheme.get("associated_scheme")}

Sponsored By:
{scheme.get("sponsored_by")}

Funding Pattern:
{scheme.get("funding_pattern")}

Beneficiaries:
{scheme.get("beneficiaries")}

Benefit Type:
{scheme.get("benefit_type")}

Eligibility:
{scheme.get("eligibility")}

How To Avail:
{scheme.get("how_to_avail")}

Validity:
{scheme.get("validity")}

Introduced On:
{scheme.get("introduced_on")}

Scheme Type:
{scheme.get("scheme_type")}

Description:
{scheme.get("description")}

Government Source:
{scheme.get("url")}
"""
            )

        return "\n".join(
            context_parts
        )

    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    def generate_answer(
        self,
        question,
        context,
        conversation_history
    ):

        history_text = (
            self.format_conversation_history(
                conversation_history
            )
        )

        prompt = f"""
You are a helpful assistant for Tamil Nadu
Government schemes.

Answer the user's current question using ONLY
the provided government scheme information.

Do not invent information.

Use previous conversation only to understand
what the user means.

If the requested information is not available
in the retrieved scheme information, clearly
say that it is not available.

For relevant schemes, provide:

- Scheme name
- Benefit type
- Beneficiary
- Short explanation

Keep the answer clear and reasonably concise.

Previous conversation:

{history_text}

Current user question:

{question}

Government scheme information:

{context}
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content

    # ========================================================
    # BUILD SOURCES
    # ========================================================

    def build_sources(
        self,
        results
    ):

        sources = []

        for result in results:

            scheme = result["scheme"]

            sources.append(
                {
                    "title": scheme.get(
                        "title"
                    ),

                    "url": scheme.get(
                        "url"
                    ),

                    "department": scheme.get(
                        "department"
                    ),

                    "district": scheme.get(
                        "district"
                    ),

                    "organisation": scheme.get(
                        "organisation"
                    ),

                    "associated_scheme": scheme.get(
                        "associated_scheme"
                    ),

                    "sponsored_by": scheme.get(
                        "sponsored_by"
                    ),

                    "funding_pattern": scheme.get(
                        "funding_pattern"
                    ),

                    "beneficiary": scheme.get(
                        "beneficiaries"
                    ),

                    "benefit_type": scheme.get(
                        "benefit_type"
                    ),

                    "eligibility": scheme.get(
                        "eligibility"
                    ),

                    "how_to_avail": scheme.get(
                        "how_to_avail"
                    ),

                    "validity": scheme.get(
                        "validity"
                    ),

                    "introduced_on": scheme.get(
                        "introduced_on"
                    ),

                    "scheme_type": scheme.get(
                        "scheme_type"
                    ),

                    "description": scheme.get(
                        "description"
                    ),

                    "uploaded_file": scheme.get(
                        "uploaded_file"
                    ),

                    "retrieved_by": result.get(
                        "retrieved_by"
                    )
                }
            )

        return sources

    # ========================================================
    # MAIN ASK
    # ========================================================

    def ask(
        self,
        question,
        conversation_history=None
    ):

        if conversation_history is None:

            conversation_history = []

        print("\n" + "=" * 70)

        print(
            f"QUESTION: {question}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # 1. Query understanding
        # ----------------------------------------------------

        filters = self.understand_query(
            question,
            conversation_history
        )

        print(
            "\nDetected filters:"
        )

        print(
            filters.model_dump()
        )

        # ----------------------------------------------------
        # 2. Vector search
        # ----------------------------------------------------

        vector_results = self.vector_search(
            question,
            k=5
        )

        print(
            f"Vector results: "
            f"{len(vector_results)}"
        )

        # ----------------------------------------------------
        # 3. Graph search
        # ----------------------------------------------------

        graph_results = self.graph_search(
            filters
        )

        graph_results = (
            self.validate_graph_results(
                graph_results,
                filters
            )
        )

        print(
            f"Graph results: "
            f"{len(graph_results)}"
        )

        # ----------------------------------------------------
        # 4. Structured search
        # ----------------------------------------------------

        structured_results = (
            self.structured_search(
                filters
            )
        )

        print(
            f"Structured results: "
            f"{len(structured_results)}"
        )

        # ----------------------------------------------------
        # 5. Combine
        # ----------------------------------------------------

        combined_results = (
            self.combine_results(
                vector_results,
                graph_results,
                structured_results
            )
        )

        print(
            f"Combined results: "
            f"{len(combined_results)}"
        )

        # ----------------------------------------------------
        # 6. Build context
        # ----------------------------------------------------

        context = self.build_context(
            combined_results
        )

        # ----------------------------------------------------
        # 7. Generate answer
        # ----------------------------------------------------

        answer = self.generate_answer(
            question,
            context,
            conversation_history
        )

        # ----------------------------------------------------
        # 8. Build sources
        # ----------------------------------------------------

        sources = self.build_sources(
            combined_results
        )

        # ----------------------------------------------------
        # 9. Return
        # ----------------------------------------------------

        return {

            "question": question,

            "filters": filters.model_dump(),

            "vector_results": len(
                vector_results
            ),

            "graph_results": len(
                graph_results
            ),

            "structured_results": len(
                structured_results
            ),

            "combined_results": len(
                combined_results
            ),

            "answer": answer,

            "sources": sources
        }