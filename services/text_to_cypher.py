"""
services/text_to_cypher.py
──────────────────────────
Text-to-Cypher service.

Translates natural language questions into Cypher queries using an LLM,
executes them against Neo4j, and synthesises a human-readable answer.
Uses LangChain's GraphCypherQAChain.
"""

import logging
import textwrap

from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

logger = logging.getLogger(__name__)


# ── Prompt templates ────────────────────────────────────────────────────────

_CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=textwrap.dedent("""\
        You are an expert Neo4j Cypher query writer for an IT Service Desk Knowledge Graph.

        Graph Schema:
        {schema}

        Node Labels     : System, Issue, Team_PIC, Resolution
        Relationship Types: MANAGED_BY, EXPERIENCES, RESOLVED_WITH

        Rules:
        - Use MATCH and RETURN only; no CREATE, DELETE, or MERGE.
        - Use case-insensitive matching where appropriate (toLower()).
        - Return meaningful property fields, not raw node objects.
        - Limit results to 10 unless the question asks for all.

        Question: {question}

        Cypher Query (no explanation, no markdown fences, just the query):
    """),
)

_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=textwrap.dedent("""\
        You are a helpful IT Service Desk assistant.
        Use the following graph query results to answer the question.
        If the results are empty, say you could not find the information.

        Graph Results:
        {context}

        Question: {question}

        Answer:
    """),
)


def ask_graph_database(query: str, llm, graph: Neo4jGraph) -> str:
    """
    Translate a natural language question to Cypher and return the answer.

    Workflow:
        1. LLM generates a Cypher query from the question + graph schema.
        2. Cypher executes against Neo4j.
        3. LLM synthesises a human-readable answer from the raw results.

    Args:
        query : Natural language question.
        llm   : Instantiated LangChain chat model.
        graph : Neo4jGraph connection object.

    Returns:
        str: Human-readable answer.
    """
    logger.info("Text-to-Cypher → '%s'", query)

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=_CYPHER_GENERATION_PROMPT,
        qa_prompt=_QA_PROMPT,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,
    )

    try:
        result = chain.invoke({"query": query})
        answer = result.get("result", "No answer generated.")

        steps = result.get("intermediate_steps", [])
        if steps:
            logger.info("Generated Cypher:\n%s", steps[0].get("query", ""))

        logger.info("Answer: %s", answer)
        return answer

    except Exception as exc:
        logger.error("Text-to-Cypher failed: %s", exc)
        return f"Error: {exc}"
