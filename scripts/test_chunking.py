import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


INPUT_FILE = "data/raw/schemes.json"


def main():

    # 1. Load the scraped schemes
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        schemes = json.load(file)


    # 2. Convert schemes into LangChain Documents
    documents = []

    for scheme in schemes:

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


    print("Original documents:", len(documents))


    # 3. Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )


    # 4. Split the documents
    chunks = text_splitter.split_documents(
        documents
    )


    print("Total chunks:", len(chunks))


    # 5. Display the first few chunks
    print("\nFirst 3 chunks")
    print("=" * 70)

    for index, chunk in enumerate(chunks[:3], start=1):

        print(f"\nChunk {index}")
        print("-" * 70)

        print(chunk.page_content)

        print("\nMetadata:")
        print(chunk.metadata)


if __name__ == "__main__":
    main()