"""
VoltIQ Fleet Advisor — LangGraph StateGraph

Builds the agent graph:

    START → llm_node → (has tool calls?) ─Yes→ tool_node → llm_node (loop)
                            │
                            No
                            ↓
                      extract_sources → END

Nodes:
  - llm_node:          Invokes the ChatOpenAI model with bound tools
  - tool_node:         Executes the tool calls the LLM requested
  - extract_sources:   Scans the final response for dataset/model attributions
"""

import logging
from typing import Dict, Any, List, Optional, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from app.config import settings
from app.agent.state import GraphState
from app.agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are VoltIQ's dedicated Industrial Fleet Electrification & Asset Intelligence Advisor. "
    "You are completely data-grounded. You must answer questions ONLY using our official datasets, "
    "trained ML models, and outputs from the tools provided to you. "
    "If information is not available in the datasets or tool outputs, clearly state that it is unavailable.\n\n"
    "Guidelines:\n"
    "- References to data should cite exact sources, such as 'Fleet Dataset', 'Battery Dataset', or 'Carbon Dataset'.\n"
    "- References to predictions must explicitly name the machine learning model used, e.g., 'LinearRegression Fleet Model' "
    "or 'GradientBoosting Battery Model'.\n"
    "- When returning predictions, format them clearly with: prediction output value, confidence level/performance metric "
    "(like R2 score or MAE as reported by the tools), and the model name.\n"
    "- Never make up vehicle IDs, SOH numbers, or carbon metrics. Only return facts returned by your tools.\n"
    "- Maintain a professional, action-oriented, and data-centric tone.\n"
    "- Always use multiple tools if necessary to give a complete, well-rounded answer."
)



def _get_llm() -> ChatOpenAI:
    """Create the ChatOpenAI instance with tools bound."""
    return ChatOpenAI(
        model=settings.openai_model_name,
        openai_api_key=settings.openai_api_key,
    )




def llm_node(state: GraphState) -> Dict[str, Any]:

    llm = _get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    logger.info(f"LLM node invoked | {len(state['messages'])} messages in state")
    ai_message = llm_with_tools.invoke(messages)

    return {"messages": [ai_message]}


def tool_node(state: GraphState) -> Dict[str, Any]:
    """
    Execute the tool calls requested by the LLM.

    Uses LangGraph's built-in ToolNode which:
      - Matches tool_call names to our ALL_TOOLS list
      - Runs them (in parallel if multiple calls)
      - Returns ToolMessage results with correct tool_call_ids
    """
    # Get the last AI message which contains the tool_calls
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    # Track which tools are being called
    called_names = [tc["name"] for tc in tool_calls]
    logger.info(f"Tool node executing: {called_names}")

    # Use LangGraph's built-in ToolNode for robust execution
    tool_executor = ToolNode(ALL_TOOLS)
    result = tool_executor.invoke(state)

    # Return both the tool messages and the names of tools called
    return {
        "messages": result["messages"],
        "tool_names_called": called_names,
    }


def extract_sources(state: GraphState) -> Dict[str, Any]:
    """
    Extract dataset/model source attributions from the final AI response
    and the list of tools that were called during this run.

    This node runs right before END to populate the `sources` and
    `final_response` channels.
    """
    # The last message should be the final AIMessage (no tool_calls)
    final_message = state["messages"][-1]
    response_text = final_message.content
    tool_names = state.get("tool_names_called", [])

    sources: List[str] = []
    lower = response_text.lower()

    # Primary: scan the LLM's response text for explicit mentions
    if "fleet dataset" in lower:
        sources.append("Fleet Dataset")
    if "battery dataset" in lower:
        sources.append("Battery Dataset")
    if "carbon dataset" in lower:
        sources.append("Carbon Dataset")
    if "charging dataset" in lower:
        sources.append("Charging Dataset")
    if "gradientboosting battery model" in lower:
        sources.append("GradientBoosting Battery Model")
    if "linearregression fleet model" in lower:
        sources.append("LinearRegression Fleet Model")

    # Fallback: infer from which tools were called
    if not sources:
        for name in tool_names:
            if "fleet" in name or "vehicle" in name:
                sources.append("Fleet Dataset")
            if "vehicle" in name:
                sources.append("LinearRegression Fleet Model")
            if "battery" in name:
                sources.extend(["Battery Dataset", "GradientBoosting Battery Model"])
            if "carbon" in name:
                sources.append("Carbon Dataset")
            if "route" in name or "charging" in name:
                sources.extend(["Fleet Dataset", "Charging Dataset"])

    return {
        "sources": list(set(sources)),
        "final_response": response_text,
    }



def should_continue(state: GraphState) -> Literal["tool_node", "extract_sources"]:
    """
    After the LLM node, check if the AI wants to call tools.
    If yes → route to tool_node. If no → route to extract_sources → END.
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "extract_sources"

def build_graph() -> StateGraph:
    """
    Construct and compile the VoltIQ Fleet Advisor StateGraph.

    Graph topology:
        START → llm_node → [should_continue?]
                               ├─ tool_calls → tool_node → llm_node (loop)
                               └─ no tools  → extract_sources → END
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract_sources", extract_sources)

    # Entry point
    graph.add_edge(START, "llm_node")

    # After LLM: decide whether to call tools or finish
    graph.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "extract_sources": "extract_sources",
        },
    )

    graph.add_edge("tool_node", "llm_node")
    graph.add_edge("extract_sources", END)

    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)

    logger.info(
        f"VoltIQ LangGraph agent compiled | "
        f"nodes=['llm_node', 'tool_node', 'extract_sources'] | "
        f"tools={[t.name for t in ALL_TOOLS]}"
    )

    return compiled_graph


fleet_graph = build_graph()



def run_query(
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Execute a user query through the LangGraph Fleet Advisor.

    Args:
        user_message: The user's natural-language query.
        chat_history: Optional list of prior {"role": ..., "message": ...} dicts.

    Returns:
        {"response": str, "sources": List[str]}
        Same shape as the old FleetAdvisorAgent.run_query() so chat.py
        needs zero changes.
    """
    # Derive a stable thread_id for MemorySaver session persistence
    if chat_history:
        first_msg = chat_history[0].get("message", "")
        thread_id = f"session-{abs(hash(first_msg)) % 10 ** 8}"
    else:
        thread_id = "default-session"

    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        f"run_query | thread={thread_id} | "
        f"msg='{user_message[:80]}{'...' if len(user_message) > 80 else ''}'"
    )

    # Build initial state
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "sources": [],
        "tool_names_called": [],
        "final_response": "",
    }

    # Invoke the compiled graph
    result = fleet_graph.invoke(initial_state, config=config)

    return {
        "response": result.get("final_response", ""),
        "sources": result.get("sources", []),
    }
