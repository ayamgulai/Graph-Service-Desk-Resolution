"""
utils/schema.py
───────────────
Neo4j schema introspection utility.
Pretty-prints node labels, relationship types, and property keys.
"""

from langchain_community.graphs import Neo4jGraph


def display_graph_schema(graph: Neo4jGraph) -> None:
    """
    Refresh and pretty-print the current Neo4j graph schema.

    Args:
        graph: Neo4jGraph connection object.
    """
    graph.refresh_schema()
    print("\n" + "=" * 60)
    print("  CURRENT NEO4J GRAPH SCHEMA")
    print("=" * 60)
    print(graph.schema)
    print("=" * 60 + "\n")
