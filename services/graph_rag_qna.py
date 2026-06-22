"""
services/graph_rag_qna.py
─────────────────────────
Graph RAG QnA service.

Answers free-form questions about the IT Service Desk Knowledge Graph
using a two-stage pipeline:
    Stage 1 — Retrieval : Multi-path Cypher queries traverse the graph.
    Stage 2 — Generation: LLM synthesises a grounded, source-cited answer.

Differs from Text-to-Cypher in that:
    - No LLM-generated Cypher (no Cypher injection risk).
    - Schema-aware template queries guarantee valid traversals.
    - Optimised for conversational, grounded Q&A.
"""

import logging
import textwrap
from typing import Optional

from langchain_community.graphs import Neo4jGraph

logger = logging.getLogger(__name__)


def _extract_keywords(question: str, llm) -> list[str]:
    """Use the LLM to extract 3-5 IT-domain keywords from the question."""
    prompt = (
        "Extract 3 to 5 concise IT-domain keywords from this question. "
        "Focus on system names, issue types, team names, or action verbs. "
        "Return ONLY a comma-separated list, no explanation.\n\n"
        f"Question: {question}"
    )
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        return [k.strip().lower() for k in raw.split(",") if k.strip()][:5]
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s — using question words.", exc)
        return [w.lower() for w in question.split() if len(w) > 4][:5]


def _cypher_queries(kw: str) -> tuple[str, str, str]:
    """Return three schema-aware Cypher queries for a given keyword."""
    # Path A: full chain (System → Issue → Resolution + Team)
    path_a = f"""
        MATCH (sys:System)-[:EXPERIENCES]->(issue:Issue)
        WHERE toLower(sys.id) CONTAINS '{kw}'
           OR toLower(issue.id) CONTAINS '{kw}'
        OPTIONAL MATCH (sys)-[:MANAGED_BY]->(team:Team_PIC)
        OPTIONAL MATCH (issue)-[:RESOLVED_WITH]->(res:Resolution)
        RETURN sys.id AS system, issue.id AS issue,
               team.id AS team_pic, res.id AS resolution
        LIMIT 5
    """
    # Path B: search by Team_PIC
    path_b = f"""
        MATCH (sys:System)-[:MANAGED_BY]->(team:Team_PIC)
        WHERE toLower(team.id) CONTAINS '{kw}'
        OPTIONAL MATCH (sys)-[:EXPERIENCES]->(issue:Issue)
        OPTIONAL MATCH (issue)-[:RESOLVED_WITH]->(res:Resolution)
        RETURN sys.id AS system, issue.id AS issue,
               team.id AS team_pic, res.id AS resolution
        LIMIT 5
    """
    # Path C: search by Resolution
    path_c = f"""
        MATCH (issue:Issue)-[:RESOLVED_WITH]->(res:Resolution)
        WHERE toLower(res.id) CONTAINS '{kw}'
        OPTIONAL MATCH (sys:System)-[:EXPERIENCES]->(issue)
        OPTIONAL MATCH (sys)-[:MANAGED_BY]->(team:Team_PIC)
        RETURN sys.id AS system, issue.id AS issue,
               team.id AS team_pic, res.id AS resolution
        LIMIT 5
    """
    return path_a, path_b, path_c


_BROAD_FALLBACK = """
    MATCH (sys:System)-[:EXPERIENCES]->(issue:Issue)
    OPTIONAL MATCH (sys)-[:MANAGED_BY]->(team:Team_PIC)
    OPTIONAL MATCH (issue)-[:RESOLVED_WITH]->(res:Resolution)
    RETURN sys.id AS system, issue.id AS issue,
           team.id AS team_pic, res.id AS resolution
    LIMIT 15
"""


def _retrieve_context(
    keywords: list[str], graph: Neo4jGraph
) -> tuple[list[dict], set]:
    """Run multi-path Cypher retrieval and return deduplicated records + sources."""
    records: list[dict] = []
    sources: set        = set()
    seen: set           = set()

    for kw in keywords:
        for cypher in _cypher_queries(kw):
            try:
                for row in graph.query(cypher):
                    key = (
                        row.get("system"), row.get("issue"),
                        row.get("team_pic"), row.get("resolution"),
                    )
                    if key not in seen:
                        seen.add(key)
                        records.append(row)
                        for field in ("system", "issue", "team_pic", "resolution"):
                            val = row.get(field)
                            if val and val != "N/A":
                                sources.add(f"{field.replace('_', ' ').title()}: {val}")
            except Exception as exc:
                logger.debug("Cypher failed for kw='%s': %s", kw, exc)

    # Broad fallback if nothing matched
    if not records:
        logger.info("No keyword matches — running broad fallback.")
        try:
            records = graph.query(_BROAD_FALLBACK)
        except Exception as exc:
            logger.warning("Broad fallback failed: %s", exc)

    return records, sources


def _format_context(records: list[dict]) -> str:
    """Format raw graph records into a readable context block for the LLM."""
    if not records:
        return "No matching records found in the knowledge graph."
    lines = []
    for i, rec in enumerate(records, start=1):
        lines.append(
            f"[Record {i}]\n"
            f"  System     : {rec.get('system',     'N/A')}\n"
            f"  Issue      : {rec.get('issue',      'N/A')}\n"
            f"  Team PIC   : {rec.get('team_pic',   'N/A')}\n"
            f"  Resolution : {rec.get('resolution', 'N/A')}"
        )
    return "\n\n".join(lines)


def graph_rag_qna(question: str, llm, graph: Neo4jGraph) -> dict:
    """
    Answer a free-form question using Graph Retrieval-Augmented Generation.

    Args:
        question : Natural language question from the user.
        llm      : Instantiated LangChain chat model.
        graph    : Neo4jGraph connection object.

    Returns:
        dict:
            "answer"  — LLM-generated answer grounded in graph facts.
            "context" — Raw graph records used as context.
            "sources" — Sorted list of cited node IDs.
    """
    logger.info("Graph RAG QnA → '%s'", question)

    # Stage 1: Retrieve
    keywords = _extract_keywords(question, llm)
    logger.info("Keywords: %s", keywords)

    records, sources = _retrieve_context(keywords, graph)
    context_text     = _format_context(records)
    logger.info("Retrieved %d graph records.", len(records))

    # Stage 2: Generate
    prompt = textwrap.dedent(f"""\
        You are a knowledgeable IT Service Desk assistant with access to a
        structured Knowledge Graph of IT systems, issues, responsible teams,
        and their resolutions.

        USER QUESTION:
        {question}

        ── KNOWLEDGE GRAPH CONTEXT ────────────────────────────────────────
        {context_text}
        ───────────────────────────────────────────────────────────────────

        Instructions:
        1. Answer using ONLY facts present in the graph context above.
        2. Cite graph data explicitly (e.g. "According to the graph...").
        3. If the graph lacks sufficient information, state what IS known.
        4. Do NOT hallucinate facts not in the context.
        5. Keep the answer concise (3-6 sentences) and professional.
        6. Mention Team PIC and Resolution steps when applicable.

        ANSWER:
    """)

    try:
        response = llm.invoke(prompt)
        answer   = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        answer = f"Failed to generate answer: {exc}"

    logger.info("Graph RAG QnA complete.")
    return {
        "answer":  answer.strip(),
        "context": records,
        "sources": sorted(sources),
    }
