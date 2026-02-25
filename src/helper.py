import os 
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch
from typing import List 
from langchain.schema import Document
from src.logger import setup_logger
from dotenv import load_dotenv

logger = setup_logger()

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")


logger.info("📂 Helper: load_pdf_files loaded")
def load_pdf_files(data: str) -> List[Document]:

    # Load PDF files
    pdf_loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_docs = pdf_loader.load()

    # Load TXT files
    txt_loader = DirectoryLoader(
        data,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    txt_docs = txt_loader.load()

    # Combine all documents
    documents = pdf_docs + txt_docs

    return documents


logger.info("🧹 Helper: filter_to_minimal_docs loaded")
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []

    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content.strip(),
                metadata=doc.metadata  # keep original metadata
            )
        )

    return minimal_docs

logger.info("✂ Helper: text_split loaded")

def text_split(minimal_docs: List[Document]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    texts_chunk = text_splitter.split_documents(minimal_docs)

    return texts_chunk

logger.info("📥 Helper: download_embeddings loaded")
def download_embeddings():
    model_name = "BAAI/bge-small-en-v1.5"

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    
    return embeddings
