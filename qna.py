"""
qna.py
───────────────────────────────────────────────────────────────────────────────
IT Service Desk Knowledge Graph — Interactive QnA Chatbot

A command-line chatbot that answers questions about the Knowledge Graph
using two retrieval strategies:

  Mode [1] Graph RAG QnA   — Structured Cypher retrieval + grounded LLM answer
                              Best for: factual, source-cited responses
  Mode [2] Text-to-Cypher  — LLM generates Cypher from NL, then executes it
                              Best for: flexible, arbitrary graph exploration

USAGE:
    python qna.py

PREREQUISITES:
    - Neo4j running with the knowledge graph already populated.
      Run 'python main.py' first to build the graph.
    - .env file configured (LLM_PROVIDER + Neo4j credentials).
───────────────────────────────────────────────────────────────────────────────
"""

import sys
import textwrap
import logging

# Force UTF-8 output to avoid Windows console encoding errors (cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Suppress verbose LangChain logs for a clean chat interface
logging.basicConfig(level=logging.WARNING)
logging.getLogger(__name__).setLevel(logging.INFO)

# ── Infrastructure ────────────────────────────────────────────────────────────
from config.settings  import load_env
from llm.factory      import get_llm
from graph.connection import get_graph

# ── Services ──────────────────────────────────────────────────────────────────
from services.graph_rag_qna  import graph_rag_qna
from services.text_to_cypher import ask_graph_database


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      IT SERVICE DESK — KNOWLEDGE GRAPH QnA ASSISTANT        ║
╠══════════════════════════════════════════════════════════════╣
║  Mode [1]: Graph RAG QnA   — grounded, source-cited answers  ║
║  Mode [2]: Text-to-Cypher  — flexible graph exploration      ║
║                                                              ║
║  Commands: mode · sources · history · help · quit            ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
┌──────────────────────────────────────────────────────────────┐
│  COMMANDS                                                    │
├──────────────────────────────────────────────────────────────┤
│  mode     → switch between Graph RAG and Text-to-Cypher      │
│  sources  → show graph sources from last Graph RAG answer    │
│  history  → list questions asked this session                │
│  help     → show this help                                   │
│  quit     → exit                                             │
├──────────────────────────────────────────────────────────────┤
│  EXAMPLE QUESTIONS:                                          │
│  • Which team handles billing issues?                        │
│  • What are common resolutions for account outages?          │
│  • List all systems managed by Technical Support.            │
│  • How are smart home integration failures resolved?         │
│  • What issues has the billing system experienced?           │
└──────────────────────────────────────────────────────────────┘
"""

MODES = {
    1: "Graph RAG QnA   (grounded, source-cited)",
    2: "Text-to-Cypher  (flexible graph exploration)",
}


def _sep(char: str = "─", width: int = 64) -> str:
    return char * width


def _print_answer(answer: str, mode: int) -> None:
    label = "GRAPH RAG ANSWER" if mode == 1 else "TEXT-TO-CYPHER ANSWER"
    print(f"\n  ┌── {label} {'─' * max(0, 54 - len(label))}")
    for line in answer.splitlines():
        for wl in (textwrap.wrap(line, width=58) or [""]):
            print(f"  │  {wl}")
    print(f"  └{'─' * 62}\n")


def _print_sources(sources: list) -> None:
    if not sources:
        print("\n  [No graph sources cited in last answer]\n")
        return
    print(f"\n  ┌── GRAPH SOURCES CITED {'─' * 40}")
    for src in sources:
        print(f"  │  • {src}")
    print(f"  └{'─' * 62}\n")


def _select_mode(current: int) -> int:
    print(f"\n  Current: [{current}] {MODES[current]}")
    print("  Select:")
    for k, v in MODES.items():
        print(f"    [{k}] {v}")
    while True:
        try:
            choice = int(input("  Enter 1 or 2: ").strip())
            if choice in MODES:
                return choice
        except ValueError:
            pass
        print("  Please enter 1 or 2.")


# ─────────────────────────────────────────────────────────────────────────────
# Chatbot loop
# ─────────────────────────────────────────────────────────────────────────────

def run_qna_chatbot() -> None:
    """Launch the interactive Knowledge Graph QnA chatbot."""
    print(BANNER)

    # Connect to LLM + Graph
    print("  Connecting to LLM and Neo4j Knowledge Graph...")
    try:
        load_env()
        llm   = get_llm()
        graph = get_graph()
        print("  [OK] Connected. Knowledge Graph is ready.\n")
    except Exception as err:
        print(f"\n  [ERROR] Could not connect: {err}")
        print(
            "  Make sure:\n"
            "    • Docker is running and neo4j-servicedesk container is started\n"
            "    • .env has correct Neo4j credentials\n"
            "    • LLM_PROVIDER and API key are valid\n"
        )
        sys.exit(1)

    mode         = 1
    last_sources = []
    history      = []

    print(f"  Mode: [{mode}] {MODES[mode]}")
    print("  Type 'help' for commands or ask your first question.\n")
    print(_sep())

    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── Commands ─────────────────────────────────────────────────────
        if cmd in ("quit", "exit", "q", "bye"):
            print("\n  Goodbye! Session ended.\n")
            break

        if cmd == "help":
            print(HELP_TEXT)
            continue

        if cmd == "mode":
            mode = _select_mode(mode)
            print(f"\n  [OK] Switched to [{mode}] {MODES[mode]}\n")
            continue

        if cmd == "sources":
            _print_sources(last_sources)
            continue

        if cmd == "history":
            if not history:
                print("\n  No questions asked yet this session.\n")
            else:
                print(f"\n  ── SESSION HISTORY ({len(history)} questions) ──")
                for i, (q, _) in enumerate(history, 1):
                    print(f"  [{i}] {q}")
                print()
            continue

        # ── Answer ───────────────────────────────────────────────────────
        print("\n  Searching Knowledge Graph...")
        try:
            if mode == 1:
                result       = graph_rag_qna(user_input, llm, graph)
                answer       = result["answer"]
                last_sources = result["sources"]
                _print_answer(answer, mode=1)
                if last_sources:
                    print(
                        f"  [Graph: {len(last_sources)} source nodes cited — "
                        "type 'sources' to see them]\n"
                    )
            else:
                answer       = ask_graph_database(user_input, llm, graph)
                last_sources = []
                _print_answer(answer, mode=2)

        except Exception as exc:
            print(f"\n  [ERROR] {exc}\n")
            answer = f"ERROR: {exc}"

        history.append((user_input, answer))
        print(_sep())


if __name__ == "__main__":
    run_qna_chatbot()
