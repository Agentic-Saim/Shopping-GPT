from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from shoppinggpt.tool.product_search import product_search_tool
from shoppinggpt.tool.policy_search import policy_search_tool
from langchain.prompts import ChatPromptTemplate


class ShoppingAgent:
    """
    Core AI Shopping Agent that orchestrates product search and policy queries.
    Uses LangChain's tool-calling agent pattern with shared conversational memory.
    """

    SYSTEM_PROMPT = """You are ShoppingGPT — an intelligent, friendly AI shopping assistant \
for a modern online fashion store.

Your responsibilities:
• Answer customer questions about products (search, availability, pricing, sizing, colors).
• Explain store policies (returns, shipping, warranties) using the policy search tool.
• Provide personalized recommendations based on customer preferences.
• Keep responses concise, helpful, and conversational.

Always respond in the same language the customer uses.
If product data is not found, gracefully suggest alternatives or ask for more details."""

    def __init__(self, llm, shared_memory: ConversationBufferMemory):
        self.llm = llm
        self.verbose = False
        self.memory = shared_memory
        self.tools = [product_search_tool, policy_search_tool]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{input}"),
            ("ai", "{agent_scratchpad}")
        ])

    def invoke(self, query: str) -> str:
        """Process a user query and return the agent's response."""
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            memory=self.memory
        )
        result = agent_executor.invoke({"input": query})
        return result["output"]
