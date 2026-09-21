import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


VECTORSTORE_DIR = "vectorstore"


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
    # 2. Create the embedding model
    # -------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


    # -------------------------------------------------
    # 3. Load the FAISS vector store
    # -------------------------------------------------

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("FAISS vector store loaded.")


    # -------------------------------------------------
    # 4. Create the retriever
    # -------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    print("Retriever created.")


    # -------------------------------------------------
    # 5. Create the OpenAI chat model
    # -------------------------------------------------

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    print("OpenAI LLM initialized.")


    # -------------------------------------------------
    # 6. Ask the user a question
    # -------------------------------------------------

    question = input(
        "\nAsk a question about Tamil Nadu schemes: "
    )


    # -------------------------------------------------
    # 7. Retrieve relevant documents
    # -------------------------------------------------

    documents = retriever.invoke(
        question
    )

    print(
        f"\nRetrieved {len(documents)} documents."
    )


    # -------------------------------------------------
    # 8. Display retrieved documents
    # -------------------------------------------------

    print("\nRetrieved Context")
    print("=" * 70)

    for index, document in enumerate(
        documents,
        start=1
    ):

        print(f"\nDocument {index}")
        print("-" * 70)

        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)


    # -------------------------------------------------
    # 9. Combine retrieved documents into context
    # -------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # -------------------------------------------------
    # 10. Create the RAG prompt
    # -------------------------------------------------

    prompt = f"""
You are a helpful assistant for Tamil Nadu Government schemes.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:

"I could not find this information in the available
Tamil Nadu Government scheme data."

Do not invent or assume information.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""


    # -------------------------------------------------
    # 11. Send the prompt to the LLM
    # -------------------------------------------------

    response = llm.invoke(
        prompt
    )


    # -------------------------------------------------
    # 12. Display the answer
    # -------------------------------------------------

    print("\nAnswer")
    print("=" * 70)

    print(response.content)


if __name__ == "__main__":
    main()