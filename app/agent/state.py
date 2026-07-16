import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[str]
    tool_names_called: list[str]
    final_response: str
