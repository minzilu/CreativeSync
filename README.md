# CreativeSync 📷🤖

**CreativeSync** is an agentic AI system designed specifically for a freelance photography studio. It automates client inquiries, equipment selection, lighting configuration advice, shoot planning, and knowledge retrieval using Retrieval-Augmented Generation (RAG).

---

## 🌟 Project Architecture

```text
CreativeSync/
├── src/
│   ├── agents/          # Autonomous agent workflows (booking, gear advisor, shoot planner)
│   ├── rag/             # RAG retrieval pipeline, chunking, and ChromaDB vector store
│   └── utils/           # Helper utilities, logger, environment loader
├── corpus/              # Knowledge base text documents (gear guides, workflows, price lists)
├── tests/               # Unit and integration test suite
├── .env.example         # Template for environment variables and API keys
├── .gitignore           # Exclusions for security secrets & Python build caches
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ installed
- Git installed

### Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/minzilu/CreativeSync.git
   cd CreativeSync
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and configure your API credentials:
   ```bash
   cp .env.example .env
   ```

---

## 🔍 RAG Pipeline Usage

To run the RAG document indexing and retrieval test:

```bash
python -m src.rag.pipeline
```

---

## 🌿 Branch Strategy

- `main`: Production-ready code and release builds.
- `feature/rag-pipeline`: Active development for document chunking, embedding, and vector database retrieval.

---

## 🔒 Security

All API keys and credentials are automatically excluded via `.gitignore`. Never commit `.env` or sensitive key files to version control.
