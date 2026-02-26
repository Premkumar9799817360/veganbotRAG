# app.py

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from src.llm_config import get_llm
from src.memory import get_memory  # ✔ our memory module
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import get_qa_prompt
from src.store_index import connect_vector_store
from src.retrieval_fusion import get_fused_documents
from src.logger import setup_logger

logger = setup_logger()
load_dotenv()

app = Flask(__name__)

# ---------------------------------------
# Initialize LLM, Prompt, Chain, Retriever
# ---------------------------------------

llm = get_llm()
logger.info("✅ LLM Loaded")

qa_prompt = get_qa_prompt()
logger.info("✅ Prompt Loaded")

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
logger.info("✅ QA Chain Created")

docsearch = connect_vector_store()
logger.info("✅ Vector Store Connected")

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
logger.info("✅ Retriever Ready")

# ---------------------------------------
# Routes
# ---------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/memory", methods=["GET"])
def check_memory():
    session_id = request.args.get("session_id", "default_session")
    memory = get_memory(session_id)

    # return jsonify([
    #     {
    #         "type": msg.type,
    #         "content": msg.content
    #     }
    #     for msg in memory.messages
    # ])
    
    # for chat_number only 
    messages = memory.messages
    chats = [
        {
            "chat_number": (i // 2) + 1,
            "human": messages[i].content if i < len(messages) else "",
            "ai": messages[i + 1].content if i + 1 < len(messages) else ""
        }
        for i in range(0, len(messages), 2)
    ]

    return render_template("memory.html", chats=chats)

@app.route("/clear-memory", methods=["POST"])
def clear_memory():
    session_id = request.args.get("session_id", "default_session")
    memory = get_memory(session_id)

    memory.clear()   # Clear all stored messages

    return jsonify({"message": "Memory cleared successfully!"})


@app.route("/ask", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"answer": "Please enter a question."})

        # ---------------------------------------
        # Load persistent memory for this session
        # You can get different session_id from user if needed
        # For learn/test we use a default singular session
        # ---------------------------------------
        session_id = data.get("session_id", "default_session")
        memory = get_memory(session_id)

        # 🟢 Save this user input to memory
        memory.add_user_message(question)

        # ---------------------------------------
        # RAG: retrieve docs using your fusion function
        # ---------------------------------------
        docs = get_fused_documents(question, llm, retriever)
        if not docs:
            return jsonify({"answer": "No relevant information found."})

        # ---------------------------------------
        # Load all previous messages
        # and convert them into prompt history text
        # ---------------------------------------
        history_messages = memory.messages
        history_text = ""
        for msg in history_messages:
            history_text += f"{msg.type}: {msg.content}\n"

        # ---------------------------------------
        # Build prompt including memory
        # ---------------------------------------
        combined_prompt = f"""
Chat History:
{history_text}

Context Docs:
{docs[:5]}

Question:
{question}
"""

        # 🧠 Invoke your QA chain with full prompt
        response = qa_chain.invoke({
            "context": docs[:5],
            "input": combined_prompt
        })

        # ---------------------------------------
        # Save LLM answer to memory
        # ---------------------------------------
        memory.add_ai_message(response)

        return jsonify({"answer": response})

    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        return jsonify({"answer": "Server error occurred."})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

