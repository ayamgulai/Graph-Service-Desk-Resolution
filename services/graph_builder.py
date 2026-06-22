"""
services/graph_builder.py
─────────────────────────
LLM Graph Builder service.

Responsibility: Load IT support ticket CSV data, use LLMGraphTransformer
to extract entities and relationships per the project schema, and persist
the resulting graph documents into Neo4j.

Graph Schema:
    Nodes         : System, Issue, Team_PIC, Resolution
    Relationships :
        (System)-[:MANAGED_BY]->(Team_PIC)
        (System)-[:EXPERIENCES]->(Issue)
        (Issue)-[:RESOLVED_WITH]->(Resolution)
"""

import os
import logging
import textwrap
from typing import Optional

import pandas as pd
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

from config import settings
from graph.connection import refresh_schema

logger = logging.getLogger(__name__)

# ── Fallback sample data ────────────────────────────────────────────────────
# Used when the CSV file cannot be found on disk.
FALLBACK_SAMPLE = [
    {
        "Body": (
            "The centralized account management portal is offline. "
            "Users cannot log in or update account settings."
        ),
        "Department": "Technical Support",
        "Priority": "high",
        "Tags": "['Account', 'Outage', 'IT', 'Disruption']",
    },
    {
        "Body": (
            "Billing statement shows duplicate charges for the last cycle. "
            "Customers are confused and requesting refunds."
        ),
        "Department": "Billing and Payments",
        "Priority": "medium",
        "Tags": "['Billing', 'Payment', 'Account', 'Documentation']",
    },
    {
        "Body": (
            "Smart home integration products are not syncing with Amazon Alexa. "
            "Customers report connection drops after firmware update."
        ),
        "Department": "Returns and Exchanges",
        "Priority": "low",
        "Tags": "['Product', 'Feature', 'Tech Support', 'Integration']",
    },
]


def _build_text_payload(row: pd.Series) -> str:
    """
    Combine the relevant CSV columns into a single text chunk
    for LLMGraphTransformer to extract entities from.
    """
    return (
        f"Ticket Description: {str(row.get('Body', '')).strip()}\n"
        f"Assigned Team (Team PIC): {str(row.get('Department', 'Unknown Team')).strip()}\n"
        f"Priority: {str(row.get('Priority', 'unknown')).strip()}\n"
        f"Tags / Keywords: {str(row.get('Tags', '[]')).strip()}\n"
    )


# ── Extraction prompt ───────────────────────────────────────────────────────
_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=textwrap.dedent("""\
        You are an expert IT Service Desk Knowledge Graph builder.

        Extract entities and relationships from the following IT support ticket text.
        Strictly follow this schema:

        NODE TYPES:
        - System      : The IT system, product, or service experiencing the problem
                        (e.g., "Account Management Portal", "Billing System").
        - Issue        : A specific problem or disruption described in the ticket
                        (e.g., "Portal Offline", "Duplicate Billing Charges").
        - Team_PIC     : The team responsible for resolution.
                        *** ALWAYS map 'Assigned Team (Team PIC)' to a Team_PIC node. ***
        - Resolution   : The recommended or inferred resolution step.
                        *** If no explicit resolution is stated, INFER one from
                        the Tags/Keywords and the issue context. ***

        RELATIONSHIP TYPES (extract only these):
        - (System)-[:MANAGED_BY]->(Team_PIC)
        - (System)-[:EXPERIENCES]->(Issue)
        - (Issue)-[:RESOLVED_WITH]->(Resolution)

        RULES:
        1. Every ticket MUST produce at least one Issue node.
        2. Every Issue MUST be linked to a Resolution (infer if needed).
        3. Map 'Assigned Team (Team PIC):' directly to a Team_PIC node.
        4. Use concise, canonical names for nodes.
        5. Include Priority as a property on Issue nodes when mentioned.

        Ticket Text:
        {input}
    """),
)


def build_graph_from_csv(
    file_path: str,
    llm,
    graph: Neo4jGraph,
    sample_size: Optional[int] = None,
) -> None:
    """
    Ingest IT support ticket CSV data and build a Knowledge Graph in Neo4j.

    Args:
        file_path   : Path to the CSV file.
        llm         : Instantiated LangChain chat model.
        graph       : Neo4jGraph connection object.
        sample_size : Max rows to process (default: SAMPLE_SIZE env var).
    """
    if sample_size is None:
        sample_size = settings.SAMPLE_SIZE()

    # Load CSV or use fallback
    if os.path.exists(file_path):
        logger.info("Loading CSV: %s (rows=%d)", file_path, sample_size)
        df = pd.read_csv(file_path, nrows=sample_size)
        logger.info("Loaded %d rows.", len(df))
    else:
        logger.warning("'%s' not found — using built-in fallback data.", file_path)
        df = pd.DataFrame(FALLBACK_SAMPLE)

    # Configure LLMGraphTransformer with schema constraints
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["System", "Issue", "Team_PIC", "Resolution"],
        allowed_relationships=["MANAGED_BY", "EXPERIENCES", "RESOLVED_WITH"],
        prompt=_EXTRACTION_PROMPT,
        node_properties=True,
        relationship_properties=False,
    )

    total_added = 0
    for idx, row in df.iterrows():
        try:
            payload   = _build_text_payload(row)
            doc       = Document(page_content=payload)

            logger.info(
                "[%d/%d] Extracting — Dept: %s | Priority: %s",
                int(idx) + 1, len(df),
                row.get("Department", "N/A"),
                row.get("Priority",   "N/A"),
            )

            graph_docs = transformer.convert_to_graph_documents([doc])
            graph.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True,
            )

            nodes = sum(len(gd.nodes) for gd in graph_docs)
            rels  = sum(len(gd.relationships) for gd in graph_docs)
            logger.info("  [OK] %d nodes, %d relationships extracted", nodes, rels)
            total_added += 1

        except Exception as exc:
            logger.error("  [ERROR] Ticket %d failed: %s", idx, exc)
            continue

    logger.info(
        "Graph build complete: %d/%d tickets processed.", total_added, len(df)
    )
    refresh_schema(graph)
