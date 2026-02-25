# src/tools/medical_formatter.py

import re


def convert_docs_to_medical_cards(docs, limit=5):
    """
    Convert RAG medical text documents
    into structured card format.
    """

    cards = []

    for doc in docs[:limit]:

        text = doc.page_content

        # Basic pattern extraction (you can improve later)
        title = text.split("\n")[0][:100]

        # Try to extract sections
        symptoms = extract_section(text, "Symptoms")
        treatment = extract_section(text, "Treatment")
        causes = extract_section(text, "Causes")

        card = {
            "title": title,
            "symptoms": symptoms,
            "causes": causes,
            "treatment": treatment
        }

        cards.append(card)

    return cards


def extract_section(text, section_name):
    pattern = rf"{section_name}:(.*?)(\n[A-Z][a-z]+:|\Z)"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()[:300]

    return "Not specified."