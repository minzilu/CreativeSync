"""
CreativeSync Multi-Agent State Machine (3-Node Architecture)

Directory: src/agents/agent_architecture.py
"""

import os
import sys
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

# Forcefully load keys from .env file overriding cached environment variables
load_dotenv(override=True)

# Debug prints to verify API key loading
print(f"Groq Key Loaded: {bool(os.getenv('GROQ_API_KEY'))}")
print(f"OpenRouter Key Loaded: {bool(os.getenv('OPENROUTER_API_KEY'))}")

# LangGraph imports 
try:
    from langgraph.graph import StateGraph, START, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    START = "START"
    END = "END"

    class StateGraph:
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


# LangChain LLM imports
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


# 1. State Schema Definition
class AgentState(TypedDict):
    client_inquiry: str
    parsed_intent: str
    proposal_draft: str
    final_proposal: str
    messages: List[Dict[str, str]]

StudioState = AgentState


# 2. LLM Initialization Helpers
def get_router_llm():
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
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if HAS_OPENROUTER and openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
        return ChatOpenAI(
            model_name="openrouter/auto",
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
        )
    # Fallback to Groq if OpenRouter key or package is not active
    if HAS_GROQ and os.getenv("GROQ_API_KEY"):
        return ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
    return None


# 3. Node 1: Router Agent
def router_node(state: AgentState) -> dict:
    inquiry = state["client_inquiry"]
    messages = state.get("messages", [])
    
    llm = get_router_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the Router Agent for CreativeSync Studio. Parse the client inquiry and extract: "
                "Shoot Type, Location, Date/Time, B2B Studio Status, Add-ons Selected, and Specific Requirements."
            ))
            human_msg = HumanMessage(content=inquiry)
            response = llm.invoke([sys_msg, human_msg])
            parsed_intent = response.content.strip()
        except Exception as e:
            error_msg = f"[API EXCEPTION in Router Agent (Groq)]: {str(e)}"
            print(error_msg)
            parsed_intent = error_msg
    else:
        error_msg = "[GROQ_API_KEY UNSET/MISSING]: GROQ_API_KEY is missing or invalid in .env file."
        print(error_msg)
        parsed_intent = error_msg

    updated_messages = list(messages) + [{"sender": "Router", "content": parsed_intent}]
    return {"parsed_intent": parsed_intent, "messages": updated_messages}


# 4. Node 2: Orchestrator Agent
def orchestrator_node(state: AgentState) -> dict:
    intent = state["parsed_intent"]
    messages = state.get("messages", [])
    
    llm = get_openrouter_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the Lead Studio Orchestrator for CreativeSync Photography Studio in Sri Lanka.\n"
                "Draft a professional photography proposal based on the parsed client intent.\n\n"
                "CRITICAL BUSINESS & PRICING RULES (LKR):\n"
                "1. RETAIL PRICING MATRIX:\n"
                "   - Commercial & Corporate: Starter LKR 35,000 (1 hr, 25-35 photos), Business LKR 65,000 (2-3 hrs, 45-65 photos), Premium LKR 95,000 - 125,000+ (4-8 hrs).\n"
                "   - Wedding & Pre-Shoots: Pre-Shoot LKR 50,000 avg, Wedding Day LKR 50,000 - 150,000+, Albums/Prints LKR 30,000+.\n"
                "   - Casual & Portraits: Mini/Casual LKR 10,000 - 25,000.\n\n"
                "2. B2B FREELANCE RULE:\n"
                "   - If 'B2B Studio Booking' is TRUE, DO NOT use retail pricing or include album/print costs.\n"
                "   - Apply a flat Freelance Photographer Day/Session Rate (e.g. LKR 25,000 - LKR 45,000 per session) instead of end-client retail pricing.\n\n"
                "3. OUTSTATION TRAVEL RULE:\n"
                "   - If location is outside Colombo, Malabe, or Kaduwela regions (e.g., Kandy, Galle, Negombo, Nuwara Eliya, Jaffna), automatically add an 'Outstation Travel & Logistics Fee' of LKR 15,000 (or LKR 100/km) to the breakdown.\n\n"
                "4. ADD-ONS CALCULATION RULE:\n"
                "   - Explicitly add exact LKR values for any selected add-ons:\n"
                "     * 24-Hour Express Edit: +LKR 10,000\n"
                "     * Raw Files Included: +LKR 15,000\n"
                "     * Drone Coverage: +LKR 15,000\n\n"
                "INSTRUCTIONS:\n"
                "- Analyze intent, determine base package / B2B rate, travel fee, and add-ons.\n"
                "- Output pricing exclusively in LKR (no USD).\n"
                "- Provide an itemized quote breakdown and calculate the exact total."
            ))
            human_msg = HumanMessage(content=f"Structured Intent:\n{intent}")
            response = llm.invoke([sys_msg, human_msg])
            proposal_draft = response.content.strip()
        except Exception as e:
            if HAS_GROQ and os.getenv("GROQ_API_KEY"):
                try:
                    groq_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
                    response = groq_llm.invoke([sys_msg, human_msg])
                    proposal_draft = response.content.strip()
                except Exception as ex:
                    error_msg = f"[API EXCEPTION in Orchestrator Agent]: {str(ex)}"
                    print(error_msg)
                    proposal_draft = f"Intent: {intent}\n\n{error_msg}"
            else:
                error_msg = f"[API EXCEPTION in Orchestrator Agent]: {str(e)}"
                print(error_msg)
                proposal_draft = f"Intent: {intent}\n\n{error_msg}"
    else:
        error_msg = "[LLM API UNSET/MISSING]: Neither OPENROUTER_API_KEY nor GROQ_API_KEY are configured in .env."
        print(error_msg)
        proposal_draft = f"Intent: {intent}\n\n{error_msg}"

    updated_messages = list(messages) + [{"sender": "Orchestrator", "content": proposal_draft}]
    return {"proposal_draft": proposal_draft, "messages": updated_messages}


# 5. Node 3: Reflection Agent
def reflection_node(state: AgentState) -> dict:
    draft = state["proposal_draft"]
    messages = state.get("messages", [])
    
    llm = get_openrouter_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the QA Reviewer for CreativeSync Studio in Sri Lanka.\n"
                "Review the draft proposal for safety, gear accuracy, and pricing math.\n"
                "CRITICAL PRICING RULE: Preserve all LKR rates, B2B rates, travel fees, and add-on costs exactly as quoted. Do NOT convert to USD ($).\n"
                "CRITICAL OUTPUT RULE: Your final output must ONLY contain the polished client-facing proposal.\n"
                "STRIP OUT all internal gear lists, camera specs, and safety audit notes."
            ))
            human_msg = HumanMessage(content=f"Draft Proposal:\n{draft}")
            response = llm.invoke([sys_msg, human_msg])
            final_proposal = response.content.strip()
        except Exception as e:
            if HAS_GROQ and os.getenv("GROQ_API_KEY"):
                try:
                    groq_llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=os.getenv("GROQ_API_KEY"))
                    response = groq_llm.invoke([sys_msg, human_msg])
                    final_proposal = response.content.strip()
                except Exception as ex:
                    error_msg = f"[API EXCEPTION in Reflection Agent]: {str(ex)}"
                    print(error_msg)
                    final_proposal = f"{draft}\n\n{error_msg}"
            else:
                error_msg = f"[API EXCEPTION in Reflection Agent]: {str(e)}"
                print(error_msg)
                final_proposal = f"{draft}\n\n{error_msg}"
    else:
        error_msg = "[LLM API UNSET/MISSING]: Neither OPENROUTER_API_KEY nor GROQ_API_KEY are configured in .env."
        print(error_msg)
        final_proposal = f"{draft}\n\n{error_msg}"

    updated_messages = list(messages) + [{"sender": "Reflection", "content": final_proposal}]
    return {"final_proposal": final_proposal, "messages": updated_messages}


# 6. LangGraph Graph Construction
def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("reflection", reflection_node)
    
    builder.add_edge(START, "router")
    builder.add_edge("router", "orchestrator")
    builder.add_edge("orchestrator", "reflection")
    builder.add_edge("reflection", END)
    
    return builder.compile()

app = build_agent_graph()