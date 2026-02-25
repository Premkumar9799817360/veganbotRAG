# src/prompt.py

from langchain_core.prompts import ChatPromptTemplate
from src.logger import setup_logger

logger = setup_logger()
logger.info("📜 Prompt file loaded")

def get_qa_prompt():

    system_prompt = """
You are a medical question-answering assistant.

Instructions:
- Use ONLY the provided context to answer the question.
- Do NOT use external knowledge.
- If the answer is not present in the context, say "I don't know based on the provided information."
- Keep the answer concise (maximum three sentences).
- Be medically accurate and professional.
- if any greeting repsoen show greeting message 
Context:
{context}
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {input}")
        ]
    )

    logger.info("✅ QA Prompt Ready")

    return prompt