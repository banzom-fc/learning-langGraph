"""
A simple agent that uses Google Gemini to chat with the user.
One Input, one Output.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from IPython.display import Image, display
import os
load_dotenv()

# print("API: ", os.getenv("GEMINI_API_KEY"))

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

#node
def chat(state: AgentState) -> AgentState:
    """
    Chat node that takes the current state, invokes the LLM, and appends the response to the messages.
    Args:
        state (AgentState): The current state containing messages.
    Returns:
        AgentState: The updated state with the new message appended.
    """
    response = llm.invoke(state["messages"])
    state["messages"].append(HumanMessage(content=response.content))
    return state

#state graph
graph = StateGraph(AgentState)
graph.add_node("chat", chat)

graph.add_edge(START, "chat")
graph.add_edge("chat", END)

agent = graph.compile()

# Save the graph to a file
image_bytes = agent.get_graph().draw_mermaid_png()
image_path = os.path.join(os.getcwd(), "AIAgents", "imgs", "agent-1-graph.png")
with open(image_path, "wb") as f:
    f.write(image_bytes)

# Run the graph with an initial state
query = input("You: ")
initial_state = AgentState(messages=[HumanMessage(content=query)])
final_state = agent.invoke(initial_state)
print("AI: ", final_state["messages"][-1].content)