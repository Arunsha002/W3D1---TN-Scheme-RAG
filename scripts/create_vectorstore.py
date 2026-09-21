import json
import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


INPUT_FILE = "data/raw/schemes.json"
VECTORSTORE_DIR = "vectorstore"


def load_documents():
    """Load schemes and create one LangChain Document per scheme."""

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        schemes = json.load(file)

    documents = []

    for scheme in schemes:

        # Keep the complete scheme together.
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
            "url": scheme.get("url"),
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata
            )
        )

    return documents


def main():

    # -------------------------------------------------
    # 1. Load environment variables
    # -------------------------------------------------

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY was not found in .env"
        )


    # -------------------------------------------------
    # 2. Load complete scheme documents
    # -------------------------------------------------

    documents = load_documents()

    print(
        "Complete scheme documents:",
        len(documents)
    )


    # -------------------------------------------------
    # 3. Create OpenAI embedding model
    # -------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    print(
        "Embedding model initialized."
    )


    # -------------------------------------------------
    # 4. Create FAISS vector store
    # -------------------------------------------------

    print(
        "\nCreating FAISS vector store..."
    )

    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )


    # -------------------------------------------------
    # 5. Save FAISS vector store
    # -------------------------------------------------

    vectorstore.save_local(
        VECTORSTORE_DIR
    )


    print(
        "\nFAISS vector store created successfully."
    )

    print(
        "Location:",
        VECTORSTORE_DIR
    )


if __name__ == "__main__":
    main()