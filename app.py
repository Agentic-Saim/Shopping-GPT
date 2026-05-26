from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
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
LLM            = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-flash")
SHARED_MEMORY  = ConversationBufferMemory(return_messages=True)
SEMANTIC_ROUTER = SemanticRouter()

app = Flask(__name__)


def handle_query(query: str) -> dict:
    """
    Route and process a user query.

    Returns a dict with keys:
      - 'response' (str): The AI-generated reply.
      - 'type' (str): The route taken — chitchat or product.
    """
    guided_route = SEMANTIC_ROUTER.guide(query)

    if guided_route == CHITCHAT_ROUTE_NAME:
        chain    = create_chitchat_chain(LLM, SHARED_MEMORY)
        response = chain.invoke({"input": query})
    elif guided_route == PRODUCT_ROUTE_NAME:
        agent    = ShoppingAgent(LLM, SHARED_MEMORY)
        response = agent.invoke(query)
    else:
        response = "I'm not sure how to help with that. Could you rephrase your question?"

    # Normalise response to a plain string
    content = (
        response.content if hasattr(response, "content")
        else response["output"] if isinstance(response, dict) and "output" in response
        else str(response)
    )

    # Persist turn in shared memory
    SHARED_MEMORY.chat_memory.add_user_message(query)
    SHARED_MEMORY.chat_memory.add_ai_message(content)

    return {"response": content, "type": guided_route}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["GET"])
def get_bot_response():
    user_message = request.args.get("msg", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a message.", "type": "error"})
    response = handle_query(user_message)
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
