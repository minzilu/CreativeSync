"""
CreativeSync Multi-Agent State Machine (3-Node Architecture)

Directory: src/agents/agent_architecture.py
"""

import os
import sys
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangGraph imports (with fallback executor for bare environments)
try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    START = "START"
    END = "END"

    class StateGraph:
        """Lightweight fallback state graph compiler when langgraph is not installed."""
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = []

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, start, end):
            self.edges.append((start, end))

        def compile(self):
            return CompiledFallbackGraph(self.nodes)

    class CompiledFallbackGraph:
        def __init__(self, nodes):
            self.nodes = nodes

        def invoke(self, state):
            current_state = dict(state)
            for node_name in ["router", "orchestrator", "reflection"]:
                if node_name in self.nodes:
                    update = self.nodes[node_name](current_state)
                    if isinstance(update, dict):
                        current_state.update(update)
            return current_state


# LangChain LLM imports (with dynamic availability checks)
try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from langchain_openai import ChatOpenAI
    HAS_OPENROUTER = True
except ImportError:
    HAS_OPENROUTER = False

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 1. State Schema Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Structured state schema exchanged between agents.
    """
    client_inquiry: str
    parsed_intent: str
    proposal_draft: str
    final_proposal: str
    messages: List[Dict[str, str]]


# Alias for backward compatibility
StudioState = AgentState


# ---------------------------------------------------------------------------
# 2. LLM Initialization Helpers
# ---------------------------------------------------------------------------

def get_router_llm():
    """Initialize Router LLM: Groq (Llama 3.1 8B)."""
    if not HAS_GROQ:
        return None
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and groq_api_key != "your_groq_api_key_here":
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=groq_api_key,
            temperature=0.2,
        )
    return None


def get_openrouter_llm():
    """Initialize Orchestrator & Reflection LLM: OpenRouter (Claude 3.5 Sonnet)."""
    if not HAS_OPENROUTER:
        return None
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
        return ChatOpenAI(
            model_name="anthropic/claude-3.5-sonnet",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
        )
    return None
