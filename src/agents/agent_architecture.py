"""
CreativeSync Multi-Agent State Machine

Directory: src/agents/agent_architecture.py

Graph Sequence:
START -> router -> location_scout -> rag_retrieval -> orchestrator -> reflection -> END
"""

import os
import sys
import json
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
            for node_name in ["router", "location_scout", "rag_retrieval", "orchestrator", "reflection"]:
                if node_name in self.nodes:
                    update = self.nodes[node_name](current_state)
                    if isinstance(update, dict):
                        current_state.update(update)
            return current_state


# LangChain LLM & Embedding imports
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
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    try:
        from langchain_community.vectorstores.faiss import FAISS
    except ImportError:
        FAISS = None

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    pass


# 1. State Schema Definition
class AgentState(TypedDict):
    client_inquiry: str
    parsed_intent: str
    location_logistics: str
    retrieved_context: str
    proposal_draft: str
    final_proposal: str
    messages: List[Dict[str, str]]

StudioState = AgentState


# 2. LLM & FAISS Initialization Helpers
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

def load_locations_db() -> dict:
    """Load location scouting database from data/locations_db.json."""
    db_path = os.path.join(os.path.dirname(__file__), "../../data/locations_db.json")
    if not os.path.exists(db_path):
        db_path = "data/locations_db.json"
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Location Scout Warning] Failed to read locations_db.json: {e}")
    return {}

def get_faiss_vectorstore():
    """Load local FAISS vector store index from data/faiss_index."""
    if FAISS is None or HuggingFaceEmbeddings is None:
        return None
    index_dir = os.path.join(os.path.dirname(__file__), "../../data/faiss_index")
    if not os.path.exists(index_dir):
        index_dir = "data/faiss_index"
    
    if os.path.exists(index_dir):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.load_local(
                index_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore
        except Exception as e:
            print(f"[FAISS RAG Warning] Failed to load FAISS index: {e}")
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


# 4. Node 2: Location & Logistics Scout Agent
def location_scout_node(state: AgentState) -> dict:
    """
    Location & Logistics Scout Agent:
    Searches data/locations_db.json for the requested city or nearby cities.
    Extracts 1-2 relevant locations with fees, best shooting times, and notes.
    """
    intent = state.get("parsed_intent", state["client_inquiry"])
    inquiry = state["client_inquiry"]
    messages = state.get("messages", [])
    print("\n--- [NODE: LOCATION SCOUT AGENT] ---")
    print(f"Scouting location logistics for intent: {intent}")

    locations_data = load_locations_db()
    matched_spots = []
    matched_city = None

    search_text = f"{intent} {inquiry}".lower()
    for city_name, spots in locations_data.items():
        if city_name.lower() in search_text:
            matched_city = city_name
            matched_spots = spots[:2]
            break

    if matched_spots and matched_city:
        spots_formatted = []
        for s in matched_spots:
            spots_formatted.append(
                f"- **{s['name']}** ({s.get('type', 'Location')})\n"
                f"  - **Permit Fee**: {s.get('fee', 'N/A')}\n"
                f"  - **Best Shooting Time**: {s.get('best_time', 'N/A')}\n"
                f"  - **Scout Notes**: {s.get('notes', 'N/A')}"
            )
        location_logistics = (
            f"### Location Logistics & Suggestions ({matched_city})\n" +
            "\n".join(spots_formatted)
        )
    else:
        location_logistics = (
            "### Location Logistics & Suggestions\n"
            "- **Location Tip**: Please check local municipal/authority permit requirements, entrance fees, and drone clearances for your requested venue. Ensure advance booking if shooting on private property or hotel grounds."
        )

    print(f"Location Logistics Output:\n{location_logistics}")
    updated_messages = list(messages) + [{"sender": "Location Scout", "content": location_logistics}]
    return {
        "location_logistics": location_logistics,
        "messages": updated_messages
    }


# 5. Node 3: FAISS RAG Retrieval Node
def rag_retrieval_node(state: AgentState) -> dict:
    """
    FAISS RAG Retrieval Node:
    Performs similarity search on the local FAISS index to retrieve the top 3 relevant chunks,
    storing the concatenated text in retrieved_context.
    """
    query = state.get("parsed_intent") or state["client_inquiry"]
    messages = state.get("messages", [])
    print("\n--- [NODE: FAISS RAG RETRIEVAL AGENT] ---")
    print(f"Retrieving top 3 FAISS knowledge chunks for query: {query}")

    retrieved_context = ""
    vectorstore = get_faiss_vectorstore()
    if vectorstore:
        try:
            docs = vectorstore.similarity_search(query, k=3)
            retrieved_context = "\n---\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"[FAISS Search Error]: {e}")
            retrieved_context = ""
    
    if not retrieved_context:
        # Fallback to reading raw knowledge_base.txt if FAISS index is uninitialized
        kb_path = os.path.join(os.path.dirname(__file__), "../../data/knowledge_base.txt")
        if not os.path.exists(kb_path):
            kb_path = "data/knowledge_base.txt"
        if os.path.exists(kb_path):
            try:
                with open(kb_path, "r", encoding="utf-8") as f:
                    retrieved_context = f.read()[:1000]
            except Exception:
                pass
        
        if not retrieved_context:
            retrieved_context = (
                "FAISS Knowledge Base: Standard photography studio guidelines, "
                "equipment packages, color grading, terms, and delivery turnaround apply."
            )

    print(f"FAISS Context Retrieved:\n{retrieved_context[:250]}...")
    updated_messages = list(messages) + [{"sender": "RAG Retrieval", "content": retrieved_context}]
    return {
        "retrieved_context": retrieved_context,
        "messages": updated_messages
    }


# 6. Node 4: Final Proposal Compiler (Orchestrator)
def orchestrator_node(state: AgentState) -> dict:
    intent = state["parsed_intent"]
    location_info = state.get("location_logistics", "")
    retrieved_info = state.get("retrieved_context", "")
    messages = state.get("messages", [])
    
    llm = get_openrouter_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the Lead Studio Orchestrator & Final Proposal Compiler for CreativeSync Photography Studio in Sri Lanka.\n"
                "Draft a professional photography proposal grounded in the provided FAISS RAG Knowledge Base Context, Location Logistics Context, and Client Inquiry.\n\n"
                "CRITICAL CLIENT DETAILS EXTRACTION RULE:\n"
                "- Explicitly extract the Client's Full Name, Email Address, and Contact/WhatsApp Phone Number from the Client Inquiry and Structured Intent.\n"
                "- Inject these real extracted client contact details directly into the final markdown proposal under a 'Client Details' section.\n"
                "- ABSOLUTELY FORBIDDEN: You MUST NEVER output literal placeholders with brackets like '[Client Name]', '[Client Email]', '[Client Phone]', '[Client]', or '[Your Name]'. Replace ALL placeholders with the actual extracted details from the input state or 'Valued Client' if unprovided.\n\n"
                "CRITICAL BUSINESS & PRICING RULES (LKR):\n"
                "1. RETAIL PRICING MATRIX:\n"
                "   - Commercial & Corporate: Starter LKR 35,000 (1 hr, 25-35 photos), Business LKR 65,000 (2-3 hrs, 45-65 photos), Premium LKR 95,000 - 125,000+ (4-8 hrs).\n"
                "   - Wedding & Pre-Shoots: Pre-Shoot LKR 50,000 avg, Wedding Day LKR 50,000 - 150,000+, Albums/Prints LKR 30,000+.\n"
                "   - Casual & Portraits: Mini/Casual LKR 10,000 - 25,000.\n\n"
                "2. B2B FREELANCE RULE:\n"
                "   - If 'B2B Studio Booking' is TRUE, DO NOT use retail pricing or include album/print costs.\n"
                "   - Apply a flat Freelance Photographer Day/Session Rate (e.g. LKR 25,000 - LKR 45,000 per session) instead of end-client retail pricing.\n\n"
                "3. OUTSTATION TRAVEL RULE:\n"
                "   - If location is outside Colombo, Malabe, or Kaduwela regions (e.g., Kandy, Galle, Negombo, Nuwara Eliya, Matara, Jaffna), automatically add an 'Outstation Travel & Logistics Fee' of LKR 15,000 (or LKR 100/km) to the breakdown.\n\n"
                "4. ADD-ONS CALCULATION RULE:\n"
                "   - Explicitly add exact LKR values for any selected add-ons:\n"
                "     * 24-Hour Express Edit: +LKR 10,000\n"
                "     * Raw Files Included: +LKR 15,000\n"
                "     * Drone Coverage: +LKR 15,000\n\n"
                "5. GROUNDING & RAG KNOWLEDGE RULE:\n"
                "   - Use the retrieved FAISS RAG context to ground all proposal details regarding gear specs, deliverable terms, and studio policies.\n\n"
                "INSTRUCTIONS:\n"
                "- Output pricing exclusively in LKR.\n"
                "- Inject actual Client Name, Email, and Phone Number directly into the proposal without bracketed placeholders.\n"
                "- Provide an itemized quote breakdown, location suggestions, grounded equipment setup, and final calculated total quote."
            ))
            human_msg = HumanMessage(
                content=(
                    f"Client Inquiry:\n{state['client_inquiry']}\n\n"
                    f"Structured Intent:\n{intent}\n\n"
                    f"Location Logistics Context:\n{location_info}\n\n"
                    f"FAISS RAG Knowledge Base Context:\n{retrieved_info}"
                )
            )
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


# 7. Node 5: Reflection Agent
def reflection_node(state: AgentState) -> dict:
    draft = state["proposal_draft"]
    retrieved_info = state.get("retrieved_context", "")
    messages = state.get("messages", [])
    
    llm = get_openrouter_llm()
    if llm:
        try:
            sys_msg = SystemMessage(content=(
                "You are the QA Reviewer for CreativeSync Studio in Sri Lanka.\n"
                "Review the draft proposal for safety, location feasibility, gear accuracy, and pricing math.\n"
                "CRITICAL CLIENT DETAILS RULE: Ensure the client's actual Name, Email, and Phone Number from the inquiry are present in the proposal. Replace ANY remaining bracketed placeholders like [Client Name], [Client Email], [Client Phone], [Client], or [Your Name] with actual values.\n"
                "CRITICAL PRICING RULE: Preserve all LKR rates, B2B rates, travel fees, and add-on costs exactly as quoted. Do NOT convert to USD ($).\n"
                "CRITICAL LOGISTICS RULE: Ensure the Location Logistics & Suggestions section is included in the output.\n"
                "CRITICAL OUTPUT RULE: Your final output must ONLY contain the polished client-facing proposal.\n"
                "STRIP OUT all internal gear lists, camera specs, and safety audit notes."
            ))
            human_msg = HumanMessage(content=f"Draft Proposal:\n{draft}\n\nGrounded Knowledge Context:\n{retrieved_info}")
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


# 8. LangGraph Graph Construction
def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("location_scout", location_scout_node)
    builder.add_node("rag_retrieval", rag_retrieval_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("reflection", reflection_node)
    
    builder.add_edge(START, "router")
    builder.add_edge("router", "location_scout")
    builder.add_edge("location_scout", "rag_retrieval")
    builder.add_edge("rag_retrieval", "orchestrator")
    builder.add_edge("orchestrator", "reflection")
    builder.add_edge("reflection", END)
    
    return builder.compile()

app = build_agent_graph()
