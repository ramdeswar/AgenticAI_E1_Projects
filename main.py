import os
from dotenv import load_dotenv
from prompt_evaluator import PromptQualityEvaluator

load_dotenv()

def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file.")

    evaluator = PromptQualityEvaluator()

    print("Prompt Quality Scoring Agent (LangChain + Gemini)")
    print("Enter a prompt to evaluate. Press Ctrl+C to exit.\n")

    while True:
        try:
            user_prompt = input("Enter prompt:\n> ").strip()

            if user_prompt.lower() in ["bye", "exit", "quit"]:
                print("Goodbye! Exiting the program.")
                break

            if not user_prompt:
                print("Please enter a non-empty prompt.\n")
                continue

            result = evaluator.evaluate(user_prompt)
            print("\n=== Evaluation Result ===\n")
            print(result)
            print("\n=========================\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
