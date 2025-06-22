"""
Agent 3: ReAct Agent - Uses Tools and Memory
"""

from typing import Annotated, Sequence, TypedDict
"""
Annotated: Used to specify types for function parameters and return values. Example: Annotated[str, "This is a string parameter"].
Sequence: A generic type for a sequence of items, to automatically handle the state updates for Sequence such as Adding new messages to chat history. Example: Sequence[int] for a sequence of integers.
TypedDict: A way to define a dictionary with specific key-value pairs and types.
"""

from dotenv import load_dotenv
load_dotenv()  
from langchain_core.messages import BaseMessage # foundation for all message types
from langchain_core.messages import ToolMessage # passes the tool response to the LLM like content and tall_call_id
from langchain_core.messages import SystemMessage # system message that provides context to the LLM
from langchain_core.tools import tool # Tool that takes in function or coroutine directly.
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode # node that runs the tools called in the last AIMessage.
from langgraph.graph.message import add_messages # Merges two lists of messages, updating existing messages by ID.

import os


# State Schema Definition
class AgentState(TypedDict):
    """
    AgentState defines the structure of the agent's state, which includes a sequence of messages.
    This state is used to maintain the conversation history and manage interactions with the LLM.
    It is a TypedDict, which allows for type checking and validation of the state structure.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

## Tool 1: Addition Tool
@tool
def add_numbers(a: float, b: float) -> float:
    """
    Adds two numbers together.

    Args:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: The sum of the two numbers.
    """
    print(f"Adding {a} and {b}")
    return a + b

tools = [
    add_numbers,
]

# DEBUG 
#print(f"DEBUG: GEMINI_API_KEY loaded: {os.getenv('GEMINI_API_KEY')}")
    
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)


# Node 1
def model_node(state: AgentState) -> AgentState:
    response = model.invoke(["You are a helpful assistant."])
    return {
        "messages": [response]
    } # this type of return can be used to update the state with new messages along with the existing ones. Because of the add_messages decorator, it will automatically merge the new messages with the existing ones in the state.