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

## Generative AI Usage

During this development, Generative AI is use for initiative development with following manual modification. The models that have been used is Claude Sonnet 4.6 and Gemini 3.1 Pro. Here is the prompt. 
```
Act as an Expert AI & Graph Database Engineer. I need to build an "IT Service Desk Knowledge Graph" pipeline. Please generate a complete, production-ready Python script (.py) using Python, LangChain, and Neo4j. Do NOT generate a Jupyter Notebook; I need a modular, executable Python file.

This project fulfills requirement: LLM Graph Builder, Text-to-Cypher, and Graph RAG.

Crucial Requirement: The LLM setup must be adjustable via environment variables (using a `.env` file). Dynamically initialize the correct LangChain chat model based on `os.getenv("LLM_PROVIDER")`. Support at least OpenAI (`langchain-openai`) and Gemini (`langchain-google-genai`) as toggles. 

Graph Schema Design:
1. Nodes: `System`, `Issue`, `Team_PIC`, `Resolution`
2. Relationships: 
   - `(System)-[:MANAGED_BY]->(Team_PIC)`
   - `(System)-[:EXPERIENCES]->(Issue)`
   - `(Issue)-[:RESOLVED_WITH]->(Resolution)`

Dataset Context:
I have a dataset (e.g., `tickets.csv`) with the following columns:
- `Body`: Unstructured text of the user's issue.
- `Department`: The assigned team.
- `Priority`: Urgency level.
- `Tags`: List of keywords.

Please structure the Python script with the following modular functions:

1. `setup_environment()`:
- Load `.env` variables.
- Return the instantiated LLM based on `LLM_PROVIDER` and the `Neo4jGraph` connection object using `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`.

2. `build_graph_from_csv(file_path, llm, graph)`:
- Load the dataset using `pandas` (use a small sample fallback if the file isn't found).
- Iterate through rows, combine `Body`, `Department`, and `Tags` into a single text payload.
- Use `LLMGraphTransformer` to extract nodes and relationships. Ensure `Team_PIC` maps to `Department`. Instruct the LLM to infer a `Resolution` based on `Tags` and the issue if missing.
- Add the extracted graph documents to the Neo4j database.

3. `ask_graph_database(query, llm, graph)`:
- Implement `GraphCypherQAChain` to translate a natural language query into Cypher and return the answer.

4. `triage_incoming_ticket(ticket_text, llm, graph)`:
- A Graph RAG function. Retrieve the relevant `System`, historical `Issue`, `Team_PIC`, and `Resolution` from the graph based on the incoming `ticket_text`.
- Pass this context to the LLM to generate an automated routing and triage response.

Execution Block:
- Create an `if __name__ == "__main__":` block that ties these functions together to demonstrate the complete pipeline (Setup -> Ingest Data -> Test Cypher QA -> Test Ticket Triage).

Requirements:
- Add clear docstrings and inline comments.
- Include a sample `.env` file structure in the script's docstring at the very top. 
``` 

Manual modification needed to make it ideal for multiple models and implement clean architecture.