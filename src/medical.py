# domains/medical.py

def format_response(query, docs):
    """
    Format medical RAG results.
    For now return plain medical explanation.
    """

    if not docs:
        return {
            "type": "text",
            "answer": "No medical information found."
        }

    # Combine top 3 medical docs
    combined_text = "\n\n".join([doc.page_content for doc in docs[:3]])

    return {
        "type": "text",
        "answer": combined_text
    }