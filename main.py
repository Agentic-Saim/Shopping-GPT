import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from shoppinggpt.router.lib_semantic_router import (
    SemanticRouter,
    PRODUCT_ROUTE_NAME,
    CHITCHAT_ROUTE_NAME
)
from shoppinggpt.chain import create_chitchat_chain
from shoppinggpt.agent import ShoppingAgent

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ── Core Components ────────────────────────────────────────────────────────────
# Switch between Gemini and Groq by uncommenting the desired LLM below:
LLM = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
# from langchain_groq import ChatGroq
# LLM = ChatGroq(temperature=0, model="llama3-8b-8192")

SHARED_MEMORY  = ConversationBufferMemory(return_messages=True)
SEMANTIC_ROUTER = SemanticRouter()


def handle_query(query: str) -> dict:
    """
    Route and process a user query through the semantic router.

    Args:
        query: Raw user input string.

    Returns:
        dict with 'response' (str) and 'type' (str) keys.
    """
    try:
        with np.errstate(invalid="ignore"):
            guided_route = SEMANTIC_ROUTER.guide(query)
    except RuntimeWarning:
        guided_route = CHITCHAT_ROUTE_NAME

    if guided_route == CHITCHAT_ROUTE_NAME:
        chain    = create_chitchat_chain(LLM, SHARED_MEMORY)
        response = chain.invoke({"input": query})
    elif guided_route == PRODUCT_ROUTE_NAME:
        agent    = ShoppingAgent(LLM, SHARED_MEMORY)
        response = agent.invoke(query)
    else:
        response = "I'm not sure how to handle that. Please try rephrasing."

    content = (
        response.content if hasattr(response, "content")
        else response["output"] if isinstance(response, dict) and "output" in response
        else str(response)
    )

    SHARED_MEMORY.chat_memory.add_user_message(query)
    SHARED_MEMORY.chat_memory.add_ai_message(content)

    return {"response": content, "type": guided_route}


def main():
    """Run ShoppingGPT in interactive CLI mode."""
    print("\n" + "=" * 55)
    print("  🛍️  Welcome to ShoppingGPT — Your AI Fashion Advisor")
    print("  Type 'exit' or 'quit' to end the session.")
    print("=" * 55 + "\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("\nThanks for shopping with ShoppingGPT. Goodbye! 👋\n")
            break

        try:
            result = handle_query(user_input)
            print(f"\nShoppingGPT [{result['type']}]: {result['response']}\n")
        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}\n")


if __name__ == "__main__":
    main()