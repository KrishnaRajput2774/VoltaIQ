"""
VoltIQ Fleet Advisor — LangGraph State Definition

Defines the GraphState TypedDict that flows through every node in the
LangGraph StateGraph. Each key is a channel that nodes can read/write.

Architecture:
    User query → LLM node → (tool calls?) → Tool node → LLM node → ... → END
                              ↕                ↕
                         GraphState carries messages, sources, metadata
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# GraphState — the single state object shared across all graph nodes
# ---------------------------------------------------------------------------

class GraphState(dict):
    """
    State schema for the VoltIQ Fleet Advisor LangGraph agent.

    Channels:
    ---------
    messages : list[BaseMessage]
        Full conversation history (System → Human → AI → Tool → AI → ...).
        Uses LangGraph's `add_messages` reducer so each node appends to
        the list rather than overwriting it.

    sources : list[str]
        Accumulated dataset and model source attributions for the current
        query (e.g. "Fleet Dataset", "GradientBoosting Battery Model").
        Uses `operator.add` reducer so nodes can contribute sources
        independently without clobbering earlier entries.

    tool_names_called : list[str]
        Names of tools that were invoked during this agent run.
        Used at the end to infer source attributions if the LLM didn't
        explicitly mention them. Uses `operator.add` reducer.

    final_response : str
        The synthesised markdown answer produced by the LLM after all
        tool calls are complete. Written by the final LLM node and
        read by the API layer to build the ChatQueryResponse.
    """
    pass


# We use Annotated types with reducers so that LangGraph knows how to
# merge updates from parallel branches and sequential nodes.
#
# add_messages:  intelligently merges message lists (handles tool_call_id dedup)
# operator.add:  simple list concatenation for sources and tool names

from typing import TypedDict


class GraphState(TypedDict):
    # Core conversation history — the backbone of any LangGraph agent.
    # add_messages handles deduplication by message ID and proper ordering
    # of HumanMessage → AIMessage → ToolMessage sequences.
    messages: Annotated[list[BaseMessage], add_messages]

    # Dataset/model attributions accumulated during the run.
    # Each tool node or the source-extraction step appends its
    # relevant sources here. Duplicates are de-duped at the end.
    sources: Annotated[list[str], operator.add]

    # Tracks which tools were actually invoked. Populated by the
    # tool-execution node each time it runs a tool. Used as a
    # fallback for source attribution if the LLM response text
    # doesn't explicitly mention dataset names.
    tool_names_called: Annotated[list[str], operator.add]

    # The final markdown response string. Set by the last LLM node
    # after all tool results have been synthesised. The API layer
    # reads this to build the ChatQueryResponse.
    final_response: str
