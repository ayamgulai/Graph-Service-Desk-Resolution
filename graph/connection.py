"""
graph/connection.py
───────────────────
Neo4j graph connection factory.
Call get_graph() to get a ready-to-use Neo4jGraph instance.
"""

import logging
from langchain_community.graphs import Neo4jGraph
from config import settings

logger = logging.getLogger(__name__)


def get_graph() -> Neo4jGraph:
    """
    Create and return a Neo4jGraph connection object.

    Uses NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD from the .env file.
    enhanced_schema is disabled to avoid requiring the APOC plugin.

    Returns:
        Neo4jGraph: Connected graph object.

    Raises:
        ValueError       : if NEO4J_PASSWORD is not set.
        ConnectionError  : if the Neo4j database is unreachable.
    """
    uri      = settings.NEO4J_URI()
    username = settings.NEO4J_USERNAME()
    password = settings.NEO4J_PASSWORD()

    if not password:
        raise ValueError("NEO4J_PASSWORD is not set in .env")

    try:
        graph = Neo4jGraph(
            url=uri,
            username=username,
            password=password,
            # Disable APOC-dependent schema introspection
            enhanced_schema=False,
            sanitize=True,
        )
        logger.info("Neo4j connected -> %s", uri)
        return graph
    except Exception as exc:
        raise ConnectionError(
            f"Failed to connect to Neo4j at '{uri}': {exc}"
        ) from exc


def refresh_schema(graph: Neo4jGraph) -> None:
    """Refresh the cached schema metadata on the graph object."""
    graph.refresh_schema()
    logger.info("Neo4j schema refreshed.")
