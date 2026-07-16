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
    "- Always use the exact model_name, data_source, and confidence values returned by the tool output — never assume "
    "or infer a model name yourself. If a tool reports 'Fallback Math Model' or a fallback status, say so plainly and "
    "explain that the trained model was unavailable for that specific result — do not present it as if the primary "
    "trained model produced it.\n"
    "- When returning predictions, format them clearly with: prediction output value, confidence level/performance metric "
    "(like R2 score or MAE as reported by the tools), and the model name.\n"
    "- If a tool returns a confidence value as 'unavailable' or 'N/A', state that confidence data could not be retrieved "
    "for that result rather than omitting it or estimating a number.\n"
    "- If a tool includes a 'confidence_note' or similar explanatory field, incorporate it when reporting high-confidence "
    "scores so the user understands why the metric is what it is.\n"
    "- Never make up vehicle IDs, SOH numbers, readiness scores, or carbon metrics. Only return facts returned by your tools.\n"
    "- Maintain a professional, action-oriented, and data-centric tone.\n"
    "- Always use multiple tools if necessary to give a complete, well-rounded answer."
    "In last Proide the summary"
)

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
llm_with_tools = llm.bind_tools(ALL_TOOLS)


def llm_node(state: GraphState) -> Dict[str, Any]:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]   
    
   
    try:
        ai_message = llm_with_tools.invoke(messages)
        logger.info(f"LLM node invoked | {len(state['messages'])} messages in state")
    except Exception as ex:
        logger.error(f"LLM call failed: {ex}", exc_info=True)
        ai_message = AIMessage(content="I hit an error reaching the model — please retry.")

    return {"messages": [ai_message]}


def tool_node(state: GraphState) -> Dict[str, Any]:

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    called_names = [tc["name"] for tc in tool_calls]
    logger.info(f"Tool node executing: {called_names}")

    tool_executor = ToolNode(ALL_TOOLS)
    result = tool_executor.invoke(state)
 
    return {
        "messages": result["messages"],
        "tool_names_called": called_names,
    }


import json

def extract_sources(state: GraphState) -> Dict[str, Any]:
    final_message = state["messages"][-1]
    response_text = final_message.content

    sources: set[str] = set()
    
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break  # stop at the start of this turn
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            if "data_source" in payload:
                sources.update(s.strip() for s in payload["data_source"].split(","))
            for key in ("soh_model_used", "rul_model_used", "model_name"):
                if key in payload:
                    sources.add(payload[key])

    return {"sources": sorted(sources), "final_response": response_text}



def should_continue(state: GraphState) -> Literal["tool_node", "extract_sources"]:

    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return "extract_sources"

def build_graph() -> StateGraph:
    
    graph = StateGraph(GraphState)

   
    graph.add_node("llm_node", llm_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("extract_sources", extract_sources)

   
    graph.add_edge(START, "llm_node")

    
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
    if chat_history:
        first_msg = chat_history[0].get("message", "")
        thread_id = f"session-{abs(hash(first_msg)) % 10 ** 8}"
    else:
        thread_id = "default-session"

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}

    logger.info(
        f"run_query | thread={thread_id} | "
        f"msg='{user_message[:80]}{'...' if len(user_message) > 80 else ''}'"
    )

 
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "sources": [],
        "tool_names_called": [],
        "final_response": "",
    }

   
    result = fleet_graph.invoke(initial_state, config=config)

    return {
        "response": result.get("final_response", ""),
        "sources": result.get("sources", []),
    }
