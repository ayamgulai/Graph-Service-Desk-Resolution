# IT Service Desk Knowledge Graph & Graph RAG Pipeline

A production-ready pipeline utilizing **LangChain**, **Neo4j**, and **OpenRouter** to ingest IT Service Desk support tickets, construct a structured knowledge graph, and perform advanced Retrieval-Augmented Generation (Graph RAG) for ticket triaging and interactive Q&A.

---

## 🏗️ Project Architecture (Clean Architecture)

The codebase is organized into modular components:

- **[main.py]**: Ingestion pipeline runner (demo entry point).
- **[qna.py]**: Interactive command-line Q&A chatbot.
- **[config/settings.py]**: Central configuration registry.
- **[llm/factory.py]**: Instantiates supported LLMs (OpenRouter, OpenAI, Gemini, etc.).
- **[graph/connection.py]**: Neo4j database connector.
- **services/**: Business logic layers
  - [graph_builder.py]: LLM-driven graph extraction & population.
  - [text_to_cypher.py]: Text-to-Cypher QA chain.
  - [graph_rag_qna.py]: Safe multi-path Graph RAG.
  - [triage.py]: Inbound ticket triage routing.

---

## 🛠️ Graph Schema

The pipeline automatically constructs a graph with the following structure:
- **Nodes**: `System`, `Issue`, `Team_PIC`, `Resolution`
- **Relationships**:
  - `(System)-[:MANAGED_BY]->(Team_PIC)`
  - `(System)-[:EXPERIENCES]->(Issue)`
  - `(Issue)-[:RESOLVED_WITH]->(Resolution)`

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure the following are installed:
- **Python 3.9+**
- **Docker Desktop**
- An **OpenRouter API Key** (from [openrouter.ai](https://openrouter.ai/))

### 2. Run Neo4j local instance
1. Open **Docker Desktop**.
2. Run the startup script to initialize the Neo4j container:
   - **Windows**:
     ```powershell
     .\start_neo4j.bat
     ```
   - **macOS / Linux**:
     ```bash
     docker run -d --name neo4j-servicedesk -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/ServiceDesk2024! neo4j:latest
     ```

### 3. Configure `.env`
1. Copy `.env.example` to `.env`:
   - **Windows**: `copy .env.example .env`
   - **macOS / Linux**: `cp .env.example .env`
2. Open `.env` and set:
   ```ini
   LLM_PROVIDER=openrouter
   OPENROUTER_API_KEY=your-openrouter-key-here
   OPENROUTER_MODEL_NAME=google/gemini-2.0-flash-exp
   NEO4J_PASSWORD=ServiceDesk2024!
   ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Ingestion Pipeline
```bash
python main.py
```

### 6. Start the Interactive Q&A Chatbot
```bash
python qna.py
```

---

## 💬 Chatbot Commands

During the `qna.py` session, use these commands:
- `mode` - Toggles between **Graph RAG** (context-grounded) and **Text-to-Cypher** (schema-translation) modes.
- `sources` - Shows the exact graph node references used for the last answer.
- `history` - Lists questions asked in the current session.
- `help` - Lists all available commands.
- `quit` - Exits the chat.