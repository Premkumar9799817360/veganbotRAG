from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from src.llm_config import get_llm 
from src.logger import setup_logger
logger = setup_logger()


# for llm 
llm  = get_llm()

# Query Generation Function 
logger.info("🔹 Multi Query Generation Loaded")

def generate_multiple_question(question):
    query_prompt  = ChatPromptTemplate.from_template(
        """You are a helpgul assistant.
        Generate 4 different seach queries related to the following question.
        Each query should be on a new line.
        Question : {input} """
    )

    query_chain = (
        query_prompt
        | llm
        | StrOutputParser()
    )

    response = query_chain.invoke({"input": question})

    # Correct newline split
    queries = [q.strip() for q in response.split("\n") if q.strip()]

    logger.info("✅ Generated Queries:")

    return queries



#Multi Query Retrival 
logger.info("🔹 Retrieval Function Loaded")
def retrieval_documents(queries, retriever):
    all_result = []
    for query in queries:
        docs  = retriever.invoke(query)
        all_result.append(docs)
    return all_result


# Reciprocal Rank Fusion code 

logger.info("🔹 RRF Fusion Loaded")
def recipocal_rank_fusion(results, k=60):

    fused_scores  = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)

            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0 
            
            fused_scores[doc_str] += 1/(k+rank+1)
    
    reranked_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    reranked_docs = [loads(doc) for doc, score in reranked_results]

    return reranked_docs

logger.info("🔹 Fusion Pipeline Ready")
def get_fused_documents(question, llm, retriever):

    queries = generate_multiple_question(question)
    results = retrieval_documents(queries, retriever)
    logger.info("📄 Fusion Retrieval Completed")
    return recipocal_rank_fusion(results)
