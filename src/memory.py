# src/memory.py

from langchain_community.chat_message_histories import SQLChatMessageHistory

# SQLite DB file will be created in project root
DB_CONNECTION = "sqlite:///chat_memory.db"

def get_memory(session_id: str):
    """
    Returns a SQLChatMessageHistory instance that
    stores and retrieves chat history for a session.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_CONNECTION
    )