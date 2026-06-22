"""
services/triage.py
──────────────────
Graph RAG Ticket Triage service.

Automatically routes and triages incoming IT support tickets by:
    1. Extracting keywords from the ticket text.
    2. Retrieving relevant historical context from the Knowledge Graph.
    3. Generating a structured triage report via the LLM.

Output sections: Ticket Summary, Affected System, Issue Category,
Priority Assessment, Routing Recommendation, Suggested Resolution,
Escalation Flag.
"""

import logging
import textwrap

from langchain_community.graphs import Neo4jGraph

logger = logging.getLogger(__name__)


def triage_incoming_ticket(ticket_text: str, llm, graph: Neo4jGraph) -> str:
    """
    Graph RAG triage for an incoming IT support ticket.

    Args:
        ticket_text : Raw ticket text from the user.
        llm         : Instantiated LangChain chat model.
        graph       : Neo4jGraph connection object.

    Returns:
        str: Structured 7-section triage report.
    """
    logger.info("Triaging incoming ticket...")

    # ── Step 1: Extract keywords ─────────────────────────────────────────
    kw_prompt = (
        "Extract 3-5 concise keywords (systems, issues, or technical terms) "
        "from the following IT support ticket. Return ONLY a comma-separated "
        "list, no explanation.\n\nTicket:\n" + ticket_text
    )
    try:
        kw_response = llm.invoke(kw_prompt)
        raw_kw = (
            kw_response.content
            if hasattr(kw_response, "content")
            else str(kw_response)
        )
        keywords = [k.strip().lower() for k in raw_kw.split(",") if k.strip()]
    except Exception as exc:
        logger.warning("Keyword extraction failed: %s", exc)
        keywords = ticket_text.lower().split()[:5]

    # ── Step 2: Retrieve graph context ───────────────────────────────────
    graph_context_parts = []
    for kw in keywords[:5]:
        cypher = f"""
            MATCH (sys:System)-[:EXPERIENCES]->(issue:Issue)
            WHERE toLower(sys.id) CONTAINS '{kw}'
               OR toLower(issue.id) CONTAINS '{kw}'
            OPTIONAL MATCH (sys)-[:MANAGED_BY]->(team:Team_PIC)
            OPTIONAL MATCH (issue)-[:RESOLVED_WITH]->(res:Resolution)
            RETURN sys.id AS system, issue.id AS issue,
                   team.id AS team_pic, res.id AS resolution
            LIMIT 5
        """
        try:
            for rec in graph.query(cypher):
                part = (
                    f"  System     : {rec.get('system',     'N/A')}\n"
                    f"  Issue      : {rec.get('issue',      'N/A')}\n"
                    f"  Team PIC   : {rec.get('team_pic',   'N/A')}\n"
                    f"  Resolution : {rec.get('resolution', 'N/A')}"
                )
                if part not in graph_context_parts:
                    graph_context_parts.append(part)
        except Exception as exc:
            logger.debug("Context retrieval failed for kw='%s': %s", kw, exc)

    # Broad fallback
    if not graph_context_parts:
        try:
            for rec in graph.query("""
                MATCH (sys:System)-[:EXPERIENCES]->(issue:Issue)
                OPTIONAL MATCH (sys)-[:MANAGED_BY]->(team:Team_PIC)
                OPTIONAL MATCH (issue)-[:RESOLVED_WITH]->(res:Resolution)
                RETURN sys.id AS system, issue.id AS issue,
                       team.id AS team_pic, res.id AS resolution
                LIMIT 10
            """):
                graph_context_parts.append(
                    f"  System     : {rec.get('system',     'N/A')}\n"
                    f"  Issue      : {rec.get('issue',      'N/A')}\n"
                    f"  Team PIC   : {rec.get('team_pic',   'N/A')}\n"
                    f"  Resolution : {rec.get('resolution', 'N/A')}"
                )
        except Exception as exc:
            logger.warning("Broad graph scan failed: %s", exc)

    # ── Step 3: Format context ───────────────────────────────────────────
    if graph_context_parts:
        context_section = "RELEVANT KNOWLEDGE GRAPH CONTEXT:\n" + "\n\n".join(graph_context_parts)
    else:
        context_section = (
            "RELEVANT KNOWLEDGE GRAPH CONTEXT:\n"
            "  No matching records found. Respond based on IT best practices."
        )

    # ── Step 4: Generate triage report ──────────────────────────────────
    triage_prompt = textwrap.dedent(f"""\
        You are an expert IT Service Desk Triage AI assistant.
        Analyse the incoming ticket and produce a structured triage report.

        ────────────────────────────────────────────────
        INCOMING TICKET:
        {ticket_text}
        ────────────────────────────────────────────────

        {context_section}
        ────────────────────────────────────────────────

        Generate a triage report with these sections:

        1. **Ticket Summary**         – One sentence summary.
        2. **Affected System**        – The IT system or service involved.
        3. **Issue Category**         – Type of issue (Outage, Billing Error, etc.).
        4. **Priority Assessment**    – Critical / High / Medium / Low + justification.
        5. **Routing Recommendation** – Which team and why.
        6. **Suggested Resolution**   – Step-by-step based on graph patterns.
        7. **Escalation Flag**        – Yes/No and reason.

        Be concise, actionable, and professional.
    """)

    try:
        response = llm.invoke(triage_prompt)
        result = response.content if hasattr(response, "content") else str(response)
        logger.info("Triage complete.")
        return result
    except Exception as exc:
        logger.error("Triage LLM call failed: %s", exc)
        return f"Triage failed: {exc}"
