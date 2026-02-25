# src/llm_config.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.logger import setup_logger


logger = setup_logger()
load_dotenv()


def get_llm():
    """
    Returns a production-ready ChatGroq LLM instance
    """

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")

    if not GROQ_API_KEY:
        raise ValueError("❌ GROQ_API_KEY not found in environment variables")

    if not GROQ_MODEL:
        raise ValueError("❌ GROQ_MODEL not found in environment variables")

    logger.info("🧠 Initializing Groq LLM...")

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.2,
        max_tokens=4096,   # safer default
        timeout=120,
        max_retries=6,
    )

    logger.info("✅ Groq LLM Ready")

    return llm