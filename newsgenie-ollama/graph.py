import os
from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from tools import llm, fetch_news, web_search

# --- State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    next_node: str
    category: str # For news category selection

# --- Node 1: Router/Classifier ---
def router_node(state: AgentState):
    """
    Analyzes the user query to decide if it's a news request, a general search, or a chat.
    """
    last_message = state['messages'][-1].content.lower()
    
    # Simple keyword-based routing for reliability
    news_keywords = ['news', 'latest', 'headlines', 'update', 'current events']
    
    if any(kw in last_message for kw in news_keywords):
        category = "general"
        for cat in ['technology', 'business', 'sports', 'entertainment', 'health', 'science']:
            if cat in last_message:
                category = cat
                break
        return {"next_node": "news_tool", "category": category}
    
    if "search" in last_message or "who is" in last_message or "what is" in last_message:
        return {"next_node": "search_tool"}
    
    return {"next_node": "chat_node"}

# --- Node 2: News Tool ---
def news_tool_node(state: AgentState):
    print("--- Routing to News Tool (Local) ---")
    category = state.get('category', 'general')
    news_content = fetch_news(category)
    
    response = f"Here are the latest {category} news updates (via GNews):\n\n{news_content}"
    return {"messages": [AIMessage(content=response)]}

# --- Node 3: Search Tool ---
def search_tool_node(state: AgentState):
    print("--- Routing to Web Search Tool (Local) ---")
    query = state['messages'][-1].content
    search_result = web_search(query)
    
    response = f"I searched the web for you. Here is what I found:\n\n{search_result}"
    return {"messages": [AIMessage(content=response)]}

# --- Node 4: General Chat ---
def chat_node(state: AgentState):
    print("--- Routing to General Chat (Local) ---")
    query = state['messages'][-1].content
    try:
        response = llm.invoke(query)
        content = response.content if hasattr(response, 'content') else str(response)
        return {"messages": [AIMessage(content=content)]}
    except Exception as exc:
        return {
            "messages": [
                AIMessage(
                    content=f"I couldn't complete that request because the local model is unavailable. Please make sure Ollama is running. Error: {exc}"
                )
            ]
        }

# --- Graph Construction ---
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("news_tool", news_tool_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("chat_node", chat_node)

workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda x: x["next_node"],
    {
        "news_tool": "news_tool",
        "search_tool": "search_tool",
        "chat_node": "chat_node"
    }
)

workflow.add_edge("news_tool", END)
workflow.add_edge("search_tool", END)
workflow.add_edge("chat_node", END)

app = workflow.compile()
