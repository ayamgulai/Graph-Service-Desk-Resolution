"""
llm/factory.py
──────────────
LLM provider factory.
Call get_llm() to get the correct LangChain chat model based on LLM_PROVIDER.

Supported providers:
  Cloud : openrouter | openai | gemini | anthropic
  Local : ollama | lmstudio
"""

import logging
from config import settings

logger = logging.getLogger(__name__)


def get_llm():
    """
    Instantiate and return the correct LangChain chat model based on
    the LLM_PROVIDER environment variable.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ImportError  : if the required provider package is not installed.
        ValueError   : if LLM_PROVIDER is unsupported or API key is missing.
    """
    provider = settings.LLM_PROVIDER()
    logger.info("Initialising LLM provider: '%s'", provider)

    # ── Cloud providers ──────────────────────────────────────────────────

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed.\nRun: pip install langchain-openai"
            )
        key = settings.OPENROUTER_API_KEY()
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set in .env")
        model = settings.OPENROUTER_MODEL_NAME()
        base_url = settings.OPENROUTER_BASE_URL()
        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=key,
            temperature=0,
            default_headers={
                "HTTP-Referer": settings.OPENROUTER_SITE_URL(),
                "X-Title": settings.OPENROUTER_APP_NAME(),
            },
        )
        logger.info("LLM ready -> OpenRouter | model=%s", model)
        return llm

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed.\nRun: pip install langchain-openai"
            )
        key = settings.OPENAI_API_KEY()
        if not key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        model = settings.OPENAI_MODEL_NAME()
        llm = ChatOpenAI(model=model, temperature=0, api_key=key)
        logger.info("LLM ready -> OpenAI | model=%s", model)
        return llm

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed.\n"
                "Run: pip install langchain-google-genai"
            )
        key = settings.GOOGLE_API_KEY()
        if not key:
            raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is not set in .env")
        model = settings.GEMINI_MODEL_NAME()
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            google_api_key=key,
            convert_system_message_to_human=True,
        )
        logger.info("LLM ready -> Gemini | model=%s", model)
        return llm

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed.\nRun: pip install langchain-anthropic"
            )
        key = settings.ANTHROPIC_API_KEY()
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set in .env")
        model = settings.ANTHROPIC_MODEL_NAME()
        llm = ChatAnthropic(model=model, temperature=0, api_key=key)
        logger.info("LLM ready -> Anthropic | model=%s", model)
        return llm

    # ── Local providers ──────────────────────────────────────────────────

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed.\nRun: pip install langchain-ollama"
            )
        base_url = settings.OLLAMA_BASE_URL()
        model    = settings.OLLAMA_MODEL_NAME()
        llm = ChatOllama(model=model, base_url=base_url, temperature=0)
        logger.info("LLM ready -> Ollama [LOCAL] | model=%s | server=%s", model, base_url)
        logger.info("TIP: Pull model first with: ollama pull %s", model)
        return llm

    if provider == "lmstudio":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed (required for LM Studio).\n"
                "Run: pip install langchain-openai"
            )
        base_url = settings.LMSTUDIO_BASE_URL()
        model    = settings.LMSTUDIO_MODEL_NAME()
        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key="lm-studio",  # LM Studio ignores this value
            temperature=0,
        )
        logger.info("LLM ready -> LM Studio [LOCAL] | model=%s | server=%s", model, base_url)
        logger.info("TIP: Start the Local Server in LM Studio before running.")
        return llm

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'.\n"
        "Valid options:\n"
        "  Cloud : openrouter | openai | gemini | anthropic\n"
        "  Local : ollama | lmstudio"
    )
