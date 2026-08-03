# CreativeSync: Automated Photography Proposal Generator
**Module:** IT41043 - Intelligent Systems (Agentic AI)

## 1. Overview
CreativeSync is an agentic AI system designed specifically for a freelance photography studio. It automates client inquiries, equipment selection, lighting configuration advice, shoot planning, and knowledge retrieval using a Retrieval-Augmented Generation (RAG) pipeline. The application grounds its outputs in a custom domain-specific knowledge base containing business terms, pricing structures, and regional location logistics across Sri Lanka.

**Live Application:** (https://creativesync-agent.streamlit.app/)

## 2. Project Architecture

```text
CreativeSync/
├── src/
│   ├── agents/          # Autonomous agent workflows (booking, proposal compiler)
│   └── utils/           # Helper utilities, logger, environment loader
├── data/                # Knowledge base text documents (knowledge_base.txt) and FAISS index
├── app.py               # Streamlit application and LangGraph orchestrator
├── build_rag.py         # Script to chunk text and generate the local FAISS vector store
├── .env.example         # Template for environment variables and API keys
├── .gitignore           # Exclusions for security secrets & Python build caches
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
 System Architecture & Agent Communication
The system utilizes LangGraph to facilitate structured communication between distinct agents. Data is passed through a shared AgentState TypedDict, which maintains the conversational context, extracted client constraints, retrieved RAG context, and the final compiled proposal.

sequenceDiagram
    participant UI as Streamlit Interface
    participant Graph as LangGraph Orchestrator
    participant Intent as Intent Extraction Node
    participant RAG as FAISS RAG Node
    participant Compiler as Proposal Compiler Agent
    
    UI->>Graph: Submit User Inquiry (Contact & Preferences)
    Graph->>Intent: Route to extraction
    Intent-->>Graph: Return structured constraints (Location, B2B status)
    Graph->>RAG: Forward inquiry for context search
    RAG-->>Graph: Return top-k semantic chunks (Pricing, Permits)
    Graph->>Compiler: Send combined state (Inquiry + Context)
    Compiler-->>Graph: Generate markdown proposal
    Graph-->>UI: Display output to client

