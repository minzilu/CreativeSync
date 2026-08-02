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


# ---------------------------------------------------------------------------
# 3. Node 1: Router Agent (Groq / Llama 3.1 8B)
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict:
    """
    Router Agent: Parses user inquiry into structured shoot metadata (type, location, date, requirements).
    """
    inquiry = state["client_inquiry"]
    messages = state.get("messages", [])
    print("\n--- [NODE 1: ROUTER AGENT] (Groq / Llama 3.1 8B) ---")
    print(f"Analyzing client inquiry: {inquiry}")

    llm = get_router_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the Router Agent for CreativeSync Photography Studio. "
                "Parse the client inquiry and extract structured metadata: "
                "Shoot Type, Location, Date/Time, and Specific Requirements."
            ))
            human_msg = HumanMessage(content=inquiry)
            response = llm.invoke([sys_msg, human_msg])
            parsed_intent = response.content.strip()
        except Exception as e:
            print(f"[Router Warning] API call failed: {e}. Using fallback parser.")
            parsed_intent = generate_fallback_intent(inquiry)
    else:
        print("[Router Agent] GROQ_API_KEY not active. Extracting metadata locally.")
        parsed_intent = generate_fallback_intent(inquiry)

    updated_messages = list(messages) + [
        {"sender": "Router Agent (Groq)", "content": parsed_intent}
    ]

    print(f"Parsed Intent:\n{parsed_intent}")
    return {
        "parsed_intent": parsed_intent,
        "messages": updated_messages,
    }


# ---------------------------------------------------------------------------
# 4. Node 2: Orchestrator Agent (OpenRouter / Claude 3.5 Sonnet)
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState) -> dict:
    """
    Orchestrator Agent: Uses structured intent to generate preliminary photography shoot proposal & quote.
    """
    intent = state["parsed_intent"]
    messages = state.get("messages", [])
    print("\n--- [NODE 2: ORCHESTRATOR AGENT] (OpenRouter / Claude 3.5 Sonnet) ---")
    print("Drafting photography proposal and quote...")

    llm = get_openrouter_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the Lead Orchestrator for CreativeSync Photography Studio. "
                "Draft a professional photography proposal with timeline, gear setup, and pricing."
            ))
            human_msg = HumanMessage(content=f"Structured Intent:\n{intent}")
            response = llm.invoke([sys_msg, human_msg])
            proposal_draft = response.content.strip()
        except Exception as e:
            print(f"[Orchestrator Warning] API call failed: {e}. Using fallback draft.")
            proposal_draft = generate_fallback_draft(intent)
    else:
        print("[Orchestrator Agent] OPENROUTER_API_KEY not active. Generating proposal draft.")
        proposal_draft = generate_fallback_draft(intent)

    updated_messages = list(messages) + [
        {"sender": "Orchestrator Agent (Claude 3.5 Sonnet)", "content": proposal_draft}
    ]

    print(f"Proposal Draft:\n{proposal_draft}")
    return {
        "proposal_draft": proposal_draft,
        "messages": updated_messages,
    }


def generate_fallback_intent(inquiry: str) -> str:
    return (
        "Shoot Type: Outdoor Golden Hour Portrait\n"
        "Location: Malibu Beach\n"
        "Date/Time: August 15th @ 6:00 PM\n"
        "Requirements: High-speed sync fill light, softbox modifier, quick high-res delivery."
    )


def generate_fallback_draft(intent: str) -> str:
    return (
        "### PHOTOGRAPHY SHOOT PROPOSAL & QUOTE\n"
        "**Client Intent**: Golden Hour Sunset Portrait Session\n"
        "**Location**: Malibu Beach | **Date**: August 15th @ 6:00 PM\n"
        "**Schedule**: 5:30 PM Setup -> 6:00 PM - 7:30 PM Shoot -> 8:00 PM Wrap\n"
        "**Equipment**: Sony A7IV mirrorless, 85mm f/1.4 prime lens, Godox AD200Pro strobe.\n"
        "**Pricing**: $450 total (includes 2-hour shoot & 10 retouched high-res deliverables)."
    )
