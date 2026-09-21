import json
from langchain_core.documents import Document


INPUT_FILE = "data/raw/schemes.json"


def main():

    # 1. Load the scraped JSON
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        schemes = json.load(file)


    print("Total schemes loaded:", len(schemes))


    # 2. Convert each scheme into a LangChain Document
    documents = []

    for scheme in schemes:

        # Build the searchable text
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


        # Metadata contains information about the source
        metadata = {
            "title": scheme.get("title"),
            "department": scheme.get("department"),
            "beneficiaries": scheme.get("beneficiaries"),
            "benefit_type": scheme.get("benefit_type"),
            "url": scheme.get("url"),
        }


        document = Document(
            page_content=page_content,
            metadata=metadata
        )

        documents.append(document)


    # 3. Display basic information
    print("Total LangChain Documents:", len(documents))

    print("\nFirst Document")
    print("=" * 70)

    print("Page Content:")
    print(documents[0].page_content)

    print("\nMetadata:")
    print(documents[0].metadata)


if __name__ == "__main__":
    main()