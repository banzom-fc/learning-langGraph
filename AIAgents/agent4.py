'''
AI Agent 4: Drafter Agent
'''
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

DOCUMENT_CONTENT = ""

# State Schema Definition
class AgentState(TypedDict):
    """
    AgentState defines the structure of the agent's state, which includes a sequence of messages.
    This state is used to maintain the conversation history and manage interactions with the LLM.
    It is a TypedDict, which allows for type checking and validation of the state structure.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

## Tool 1: Update Document Tool
@tool
def update_document(content: str) -> str:
    """
    Updates the document content with the provided text.

    Args:
        content (str): The text to be added to the document.

    Returns:
        str: Confirmation message indicating the document has been updated.
    """
    global DOCUMENT_CONTENT
    DOCUMENT_CONTENT += content + "\n"
    return f"Document updated with: {content}"

## Tool 2: Get Document Content Tool
@tool
def get_document_content() -> str:
    """
    Retrieves the current content of the document.

    Returns:
        str: The current content of the document.
    """
    return DOCUMENT_CONTENT if DOCUMENT_CONTENT else "Document is empty."

## Tool 3: Save Document Tool
@tool
def save_document(filename: str) -> str:
    """
    Saves the current document content to a txt file.

    Args:
        filename (str): The name of the file to save the document content.

    Returns:
        str: Confirmation message indicating the document has been saved.
    
    Example:
        >>> save_document("my_document.txt")
        Document saved to my_document.txt.
        
    Raises:
        Exception: If there is an error during file writing.
    """
    global DOCUMENT_CONTENT
    if not DOCUMENT_CONTENT:
        return "Document is empty. Nothing to save."
    
    try:
        with open(filename, 'w') as file:
            file.write(DOCUMENT_CONTENT)
        return f"Document saved to {filename}."
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
   
tools = [
    update_document,
    get_document_content,
    save_document
]
 
# Initialize the Google Generative AI model
model = ChatGoogleGenerativeAI(
    model_name="gemini-1.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)


## Node 1: Init Node
def init_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update_document' tool with the complete updated content.
    - If the user wants to view the current document content, use the 'get_document_content' tool.
    - If the user wants to save and finish, you need to use the 'save_document' tool.
    - Make sure to always show the current document state after modifications.
    - Always respond in a friendly and helpful manner.
    - If the user asks for help, provide clear instructions on how to use the tools.
    - If the user asks for the current document content, provide it in a clear and concise manner.
    - If the user asks to save the document, confirm the filename and save the content.
    - If the user asks to finish, confirm that the document has been saved and end the conversation.
    - Always remember to keep the conversation engaging and interactive.
    """)

    return {
        "messages": [system_prompt]
    }

## Node 2: Chat Node
def chat_node(state: AgentState) -> AgentState:
    query = input("User: ")
    human_message = HumanMessage(content=query)
    
    {
        "messages": [human_message]
    }
    
    response = model.invoke(state['messages'])
    ai_message = AIMessage(content=response.content)
    
    return {
        "messages": [ai_message]
    }
    

## Condition Node
def condition_node(state: AgentState) -> str:
    """
    Checks if the save document tool was called in last messages. 
    If it was, it returns "end" to indicate that the conversation should end.
    If not, it returns "continue_with_tool" to indicate that the tool should be executed
    """
    last_message = state["messages"][-1]  # Get the last message in the sequence
    
    if isinstance(last_message, ToolMessage) and last_message.tool_calls:
        return "end"
    else:
        return "continue_with_tool"

        

    
