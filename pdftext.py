from flask import Flask,render_template,request,jsonify
from dotenv import load_dotenv
import os

from src.llm_config import get_llm
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.prompt import get_qa_prompt
from src.store_index import connect_vector_store   # ✅ corrected
from src.retrieval_fusion import get_fused_documents
from src.logger import setup_logger

logger = setup_logger()
load_dotenv()

app = Flask(__name__)



@app.route("/")
def index():
    return render_template('index.html')



llm = get_llm()
logger.info("✅ LLM Loaded")


qa_prompt = get_qa_prompt()
logger.info("✅ Prompt Loaded")

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
logger.info("✅ QA Chain Created")


docsearch = connect_vector_store()   # ✅ only connect
logger.info("✅ Vector Store Connected")


retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
logger.info("✅ Retriever Ready")


@app.route("/ask", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        question = data["question"]

        docs = get_fused_documents(question, llm, retriever)
        if not docs:
            return jsonify({"answer": "No relevant information found."})

        response = qa_chain.invoke(
            {
                "context": docs[:5],
                "input": question
            }
        )
        return jsonify({"answer": response})

    except Exception as e:
        return jsonify({"answer": "Server error occurred."})
    
# -----------------------------
# TEST QUESTION
# -----------------------------

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8080, debug=True)