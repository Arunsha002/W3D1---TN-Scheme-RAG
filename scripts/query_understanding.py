from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Structured output model
# --------------------------------------------------

class QueryFilters(BaseModel):
    beneficiary: str | None = Field(
        default=None,
        description="The beneficiary mentioned in the question."
    )

    benefit_type: str | None = Field(
        default=None,
        description="The type of benefit mentioned in the question."
    )


# --------------------------------------------------
# Understand the user question
# --------------------------------------------------

def understand_question(question):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    structured_llm = llm.with_structured_output(
        QueryFilters
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a query understanding component for a
Tamil Nadu Government scheme search system.

Extract the beneficiary and benefit type from
the user's question.

Use only concepts that are actually mentioned
or clearly implied by the question.

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
benefit_type = None

Question:
Which schemes provide grants?

Output:
beneficiary = None
benefit_type = Grants

Question:
Which schemes are available for students?

Output:
beneficiary = Students
benefit_type = None

Do not invent values.
If a value cannot be determined, return None.
"""
            ),
            (
                "human",
                """
User question:

{question}
"""
            )
        ]
    )

    messages = prompt.format_messages(
        question=question
    )

    result = structured_llm.invoke(
        messages
    )

    return result


# --------------------------------------------------
# Test
# --------------------------------------------------

def main():

    load_dotenv()

    print("=" * 70)
    print("QUERY UNDERSTANDING")
    print("=" * 70)

    question = (
        "Which schemes provide subsidies to farmers?"
    )

    print("\nUser question:")
    print(question)

    result = understand_question(
        question
    )

    print("\nExtracted filters:")
    print(
        "Beneficiary:",
        result.beneficiary
    )

    print(
        "Benefit Type:",
        result.benefit_type
    )

    print("\n" + "=" * 70)
    print("QUERY UNDERSTANDING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()