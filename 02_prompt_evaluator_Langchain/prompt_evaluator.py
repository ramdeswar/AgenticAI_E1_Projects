from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


SYSTEM_EVALUAT = """
You are a strict prompt quality evaluator.

You MUST:
- Read the user's prompt.
- Score it on each of the following criteria from 0 to 10 (integers only):
  1. Clarity
  2. Specificity / Details
  3. Context
  4. Output Format & Constraints
  5. Persona Defined
- Compute the final score as the average of the five criteria (round to 1 decimal).
- Provide:
  - A short explanation (2–4 sentences)
  - 2–3 suggestions to improve the prompt

Return the answer as a **clean text report**, NOT JSON.

FORMAT EXACTLY LIKE THIS:

Final Score: X/10

Clarity: X/10
Specificity / Details: X/10
Context: X/10
Output Format & Constraints: X/10
Persona Defined: X/10

Explanation:
- sentence 1
- sentence 2

Suggestions:
1. suggestion
2. suggestion
3. suggestion (optional)

When the user enters bye then terminate the program safely
"""

EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_EVALUAT),
        ("human", "Evaluate this prompt:\n\n{user_prompt}"),
    ]
)


class PromptQualityEvaluator:
    def __init__(self, model_name="gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
        )
        self.chain = EVALUATION_PROMPT | self.llm | StrOutputParser()

    def evaluate(self, prompt: str) -> str:
        return self.chain.invoke({"user_prompt": prompt})
