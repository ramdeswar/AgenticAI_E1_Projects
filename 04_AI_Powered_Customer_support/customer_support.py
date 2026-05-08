from typing import TypedDict, List, Optional
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

class SupportState(TypedDict):
    email_text: str
    urgency: Optional[str]          # "Low" | "Medium" | "High"
    topic: Optional[str]            # "Account" | "Billing" | "Bug" | "Feature request" | "Technical issue"
    retrieved_docs: List[Document]
    response_draft: Optional[str]
    auto_reply: bool
    escalate: bool
    follow_up_action: Optional[str] # e.g. "Schedule follow-up in 3 days", "None"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM: Gemini
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY environment variable")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Load SOP / KB docs from ./sop_docs
loader = DirectoryLoader(
    "./sop_docs",
    glob="**/*.txt",
    loader_cls=lambda path: TextLoader(path, encoding="utf-8")
)
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
docs = splitter.split_documents(raw_docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


classification_prompt = ChatPromptTemplate.from_template("""
You MUST return ONLY valid JSON. No explanations.

Classify the email into:
- urgency: one of ["Low", "Medium", "High"]
- topic: one of ["Account", "Billing", "Bug", "Feature request", "Technical issue"]

Email:
{email}

Return JSON exactly like this:
{{"urgency": "Low", "topic": "Account"}}
""")

def classify_email(state):
    chain = classification_prompt | llm
    result = chain.invoke({"email": state["email_text"]})

    raw = result.content.strip()

    # --- JSON SAFE PARSER ---
    try:
        data = json.loads(raw)
    except:
        # Try to extract JSON substring
        import re
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            # fallback default
            data = {"urgency": "Medium", "topic": "Technical issue"}

    state["urgency"] = data["urgency"]
    state["topic"] = data["topic"]
    return state


def retrieve_docs(state: SupportState) -> SupportState:
    query = f"Customer email: {state['email_text']}\nTopic: {state['topic']}"
    docs = retriever.invoke(query)
    state["retrieved_docs"] = docs
    return state


response_prompt = ChatPromptTemplate.from_template("""
You are a helpful, concise customer support agent.

Customer email:
{email}

Urgency: {urgency}
Topic: {topic}

Relevant documentation:
{docs}

Write a clear, empathetic reply. 
If you need to ask for more info, do so politely.
""")

def generate_response(state: SupportState) -> SupportState:
    docs_text = "\n\n".join([d.page_content for d in state["retrieved_docs"]])
    chain = response_prompt | llm
    result = chain.invoke({
        "email": state["email_text"],
        "urgency": state["urgency"],
        "topic": state["topic"],
        "docs": docs_text or "No docs found."
    })
    state["response_draft"] = result.content
    return state


def decide_escalation(state: SupportState) -> SupportState:
    # Example rules:
    # - High urgency Billing or complex Technical issue => escalate
    # - If no docs retrieved => escalate
    high_urgency = state["urgency"] == "High"
    topic = state["topic"]
    no_docs = len(state["retrieved_docs"]) == 0

    escalate = False
    if high_urgency and topic in ["Billing", "Technical issue"]:
        escalate = True
    if no_docs:
        escalate = True

    state["escalate"] = escalate
    state["auto_reply"] = not escalate
    return state


def decide_follow_up(state: SupportState) -> SupportState:
    topic = state["topic"]
    urgency = state["urgency"]

    follow_up = "None"

    # Example rules:
    if topic in ["Bug", "Technical issue"]:
        follow_up = "Schedule follow-up in 3 days to check on fix."
    elif topic == "Billing" and urgency == "High":
        follow_up = "Follow up in 1 day to confirm billing resolution."

    state["follow_up_action"] = follow_up
    return state

graph = StateGraph(SupportState)

graph.add_node("classify", classify_email)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("generate_response", generate_response)
graph.add_node("decide_escalation", decide_escalation)
graph.add_node("decide_follow_up", decide_follow_up)

graph.set_entry_point("classify")

graph.add_edge("classify", "retrieve_docs")
graph.add_edge("retrieve_docs", "generate_response")
graph.add_edge("generate_response", "decide_escalation")
graph.add_edge("decide_escalation", "decide_follow_up")
graph.add_edge("decide_follow_up", END)

support_app = graph.compile()


def run_email(email_text: str) -> SupportState:
    initial_state: SupportState = {
        "email_text": email_text,
        "urgency": None,
        "topic": None,
        "retrieved_docs": [],
        "response_draft": None,
        "auto_reply": False,
        "escalate": False,
        "follow_up_action": None,
    }
    final_state = support_app.invoke(initial_state)
    return final_state



user_email = input("Paste the customer email:\n")

result = run_email(user_email)
print(result)

if __name__ == "__main__":
    email_text = input("Enter the customer email:\n")
    output = run_email(email_text)

    print("\n=== CLASSIFICATION ===")
    print("Urgency:", output["urgency"])
    print("Topic:", output["topic"])

    print("\n=== RESPONSE DRAFT ===")
    print(output["response_draft"])

    print("\n=== DECISION ===")
    print("Auto Reply:", output["auto_reply"])
    print("Escalate:", output["escalate"])

    print("\n=== FOLLOW UP ===")
    print(output["follow_up_action"])
