import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


# 1. Load variables from .env
load_dotenv()


# 2. Check that the API key exists
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Please check your .env file."
    )


print("OpenAI API key loaded successfully.")


# 3. Create the embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


# 4. Test embedding one piece of text
text = "Tamil Nadu Government schemes for farmers"

vector = embeddings.embed_query(text)


# 5. Display information about the vector
print("\nEmbedding created successfully.")

print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))

print("\nFirst 10 values:")
print(vector[:10])