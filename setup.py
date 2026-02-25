from setuptools import find_packages, setup

setup(
    name="Medical_chatbot",
    version="0.1.0",
    author="Prem Kumar AI Engineer",
    author_email="prembhargav2002@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain==0.3.26",
        "langchain-core==0.3.66",
        "langchain-community==0.3.26",
        "langchain-groq==0.2.3",
        "flask==3.1.1",
        "sentence-transformers==4.1.0",
        "pypdf==5.6.1",
        "python-dotenv==1.1.0",
        "huggingface-hub==0.24.6",
        "qdrant-client==1.9.1",
        "groq==0.9.0"
    ]
)