import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


VECTORSTORE_DIR = "vectorstore"


def main():

    # 1. Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY was not found in .env"
        )


    # 2. Create the same embedding model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    # 3. Load the existing FAISS vector store
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )


    print("FAISS vector store loaded successfully.")


    # 4. Ask a test question
    question = "Which schemes provide subsidies to farmers?"


    print("\nQuestion:")
    print(question)


    # 5. Search for the most relevant chunks
    results = vectorstore.similarity_search(
        question,
        k=3
    )


    # 6. Display the results
    print("\nRetrieved documents:")
    print("=" * 70)


    for index, document in enumerate(
        results,
        start=1
    ):

        print(f"\nResult {index}")
        print("-" * 70)

        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)


if __name__ == "__main__":
    main()