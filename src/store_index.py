import os
from pinecone import Pinecone, ServerlessSpec
<<<<<<< HEAD
from langchain_pinecone import Pinecone as PineconeVectorStore
=======
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
>>>>>>> 50c8da3ae75ca0de51fc94df1944981554d072a5
from dotenv import load_dotenv

from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings,
)
from src.logger import setup_logger
logger = setup_logger()

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# -----------------------------
# STEP 1: CREATE OR GET INDEX
# -----------------------------
def create_or_get_index(index_name="veganbot"):
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if index_name not in existing_indexes:
        logger.info("🆕 Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    else:
        logger.info("✅ Index already exists")

    return index_name


# -----------------------------
# STEP 2: RUN ONLY ONCE (INDEXING)
# -----------------------------
def index_documents(data_path="data/", index_name="veganbot"):
    logger.info("📂 Loading and preparing data...")

    extracted_data = load_pdf_files(data=data_path)
    filtered_data = filter_to_minimal_docs(extracted_data)
    texts_chunk = text_split(filtered_data)
    embeddings = download_embeddings()

    create_or_get_index(index_name)

    logger.info("📤 Uploading documents to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    PineconeVectorStore.from_documents(
        documents=texts_chunk,
        embedding=embeddings,
        index_name=index_name,
    )

    logger.info("✅ Indexing completed successfully.")


# -----------------------------
# STEP 3: APP MODE (NO DUPLICATE INSERT)
# -----------------------------

def connect_vector_store(index_name="veganbot"):
    logger.info("🔗 Connecting to existing index...")

    embeddings = download_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

    logger.info("✅ Connected to Pinecone")

    return vectorstore
