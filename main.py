"""
main.py
───────────────────────────────────────────────────────────────────────────────
IT Service Desk Knowledge Graph — Pipeline Demo Runner

Entry point that demonstrates the full pipeline end-to-end:
    Phase 1 : Environment setup (LLM + Neo4j)
    Phase 2 : Ingest CSV → Build Knowledge Graph  (LLM Graph Builder)
    Phase 3 : Natural language queries             (Text-to-Cypher)
    Phase 4 : Automated ticket triage             (Graph RAG Triage)

Run:
    python main.py

For interactive Q&A mode, run:
    python qna.py
───────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import logging
import textwrap

# ── Shared config ────────────────────────────────────────────────────────────
from config.settings import load_env

# ── Infrastructure factories ─────────────────────────────────────────────────
from llm.factory   import get_llm
from graph.connection import get_graph

# ── Services ─────────────────────────────────────────────────────────────────
from services.graph_builder  import build_graph_from_csv
from services.text_to_cypher import ask_graph_database
from services.triage         import triage_incoming_ticket

# ── Utilities ────────────────────────────────────────────────────────────────
from utils.schema import display_graph_schema

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the complete IT Service Desk Knowledge Graph pipeline."""

    print("\n" + "=" * 70)
    print("  IT SERVICE DESK KNOWLEDGE GRAPH — PIPELINE DEMO")
    print("=" * 70)

    # ── PHASE 1: Setup ───────────────────────────────────────────────────────
    print("\n[PHASE 1] Loading environment and connecting services...")
    try:
        load_env()
        llm   = get_llm()
        graph = get_graph()
        print("  [OK] LLM and Neo4j graph connection ready.")
    except (ValueError, ConnectionError, ImportError) as err:
        logger.critical("Setup failed: %s", err)
        sys.exit(1)

    # ── PHASE 2: Graph Builder ───────────────────────────────────────────────
    print("\n[PHASE 2] Building Knowledge Graph from CSV...")
    CSV_PATH = os.path.join("data", "it_support_ticket.csv")
    build_graph_from_csv(
        file_path=CSV_PATH,
        llm=llm,
        graph=graph,
    )
    print("  [OK] Knowledge Graph ingestion complete.")
    display_graph_schema(graph)

    # ── PHASE 3: Text-to-Cypher QA ──────────────────────────────────────────
    print("\n[PHASE 3] Text-to-Cypher QA Demo...")

    sample_queries = [
        "Which team is responsible for handling account management issues?",
        "What are the most common resolutions for billing problems?",
        "List all systems that have experienced an outage.",
    ]

    for i, query in enumerate(sample_queries, start=1):
        print(f"\n  Query {i}: {query}")
        print("  " + "-" * 60)
        answer = ask_graph_database(query=query, llm=llm, graph=graph)
        print(f"  Answer: {answer}")

    # ── PHASE 4: Graph RAG Triage ────────────────────────────────────────────
    print("\n[PHASE 4] Graph RAG Ticket Triage Demo...")

    incoming_ticket = (
        "Hello, I'm unable to access the customer portal since this morning. "
        "When I try to log in, I get an error saying 'Service Unavailable'. "
        "This is urgent as our entire team is blocked from submitting orders. "
        "We are in the Sales department and need this resolved as soon as possible."
    )

    print("\n  Incoming Ticket:")
    print("  " + "-" * 60)
    for line in textwrap.wrap(incoming_ticket, width=65):
        print(f"  {line}")
    print("  " + "-" * 60)

    triage_report = triage_incoming_ticket(
        ticket_text=incoming_ticket,
        llm=llm,
        graph=graph,
    )

    print("\n  TRIAGE REPORT:")
    print("  " + "=" * 60)
    for line in triage_report.splitlines():
        print(f"  {line}")
    print("  " + "=" * 60)

    print("\n[COMPLETE] All pipeline phases executed successfully.")
    print("Run 'python qna.py' to start the interactive Q&A chatbot.\n")


if __name__ == "__main__":
    main()
