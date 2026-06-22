"""
config/settings.py
──────────────────
Centralised configuration loader.
Loads .env variables and exposes typed config constants.
Does NOT instantiate LLM or graph — that is done by their factories.
"""

import os
from dotenv import load_dotenv


def load_env() -> None:
    """Load variables from the .env file into os.environ."""
    load_dotenv()


# ── LLM ──────────────────────────────────────────────────────────
LLM_PROVIDER          = lambda: os.getenv("LLM_PROVIDER",          "gemini").strip().lower()

# OpenAI
OPENAI_API_KEY        = lambda: os.getenv("OPENAI_API_KEY",        "")
OPENAI_MODEL_NAME     = lambda: os.getenv("OPENAI_MODEL_NAME",     "gpt-4o")

# OpenRouter
OPENROUTER_API_KEY    = lambda: os.getenv("OPENROUTER_API_KEY",    "")
OPENROUTER_MODEL_NAME = lambda: os.getenv("OPENROUTER_MODEL_NAME", "google/gemini-2.0-flash-exp")
OPENROUTER_BASE_URL   = lambda: os.getenv("OPENROUTER_BASE_URL",   "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL   = lambda: os.getenv("OPENROUTER_SITE_URL",   "http://localhost")
OPENROUTER_APP_NAME   = lambda: os.getenv("OPENROUTER_APP_NAME",   "IT-ServiceDesk-KnowledgeGraph")

# Gemini
GOOGLE_API_KEY        = lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME     = lambda: os.getenv("GEMINI_MODEL_NAME",     "gemini-2.0-flash")

# Anthropic
ANTHROPIC_API_KEY     = lambda: os.getenv("ANTHROPIC_API_KEY",     "")
ANTHROPIC_MODEL_NAME  = lambda: os.getenv("ANTHROPIC_MODEL_NAME",  "claude-3-5-sonnet-20241022")

# Ollama (local)
OLLAMA_BASE_URL       = lambda: os.getenv("OLLAMA_BASE_URL",       "http://localhost:11434")
OLLAMA_MODEL_NAME     = lambda: os.getenv("OLLAMA_MODEL_NAME",     "llama3.2")

# LM Studio (local)
LMSTUDIO_BASE_URL     = lambda: os.getenv("LMSTUDIO_BASE_URL",     "http://localhost:1234/v1")
LMSTUDIO_MODEL_NAME   = lambda: os.getenv("LMSTUDIO_MODEL_NAME",   "local-model")

# ── Neo4j ─────────────────────────────────────────────────────────
NEO4J_URI             = lambda: os.getenv("NEO4J_URI",             "bolt://localhost:7687")
NEO4J_USERNAME        = lambda: os.getenv("NEO4J_USERNAME",        "neo4j")
NEO4J_PASSWORD        = lambda: os.getenv("NEO4J_PASSWORD",        "")

# ── Pipeline ──────────────────────────────────────────────────────
SAMPLE_SIZE           = lambda: int(os.getenv("SAMPLE_SIZE",       "10"))
