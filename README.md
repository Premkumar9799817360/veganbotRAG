# 🌱 VeganChatbotRAG

> **An AI-powered, plant-based intelligent assistant** built with RAG Fusion, Reciprocal Rank Fusion, memory management, and advanced prompt engineering — deployed on Render Cloud.

---

## 📸 Project Screenshots

### 🤖 Main Chat Interface
![VeganAI Chat Interface](https://github.com/Premkumar9799817360/veganbotRAG/blob/main/image/Screenshot%202026-02-28%20220414.png)
*VeganAI — Smart Vegan Assistant. Ask anything about plant-based living: recipes, nutrition, ingredients, meal plans, and more.*

---

### 🛠️ Main Chat Interface
![Chat with Veganbot](https://github.com/Premkumar9799817360/veganbotRAG/blob/main/image/Screenshot%202026-02-28%20220502.png)
*VeganAI — Smart Vegan Assistant. Ask anything about plant-based living: recipes, nutrition, ingredients, meal plans, and more.*

---

### 🧠 Memory History & Management
![Memory History Page](https://github.com/Premkumar9799817360/veganbotRAG/blob/main/image/Screenshot%202026-02-28%20220516.png)
*Memory History page displaying all previous conversations with the ability to clear memory and navigate back to chat.*

---

## 🧭 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [RAG Fusion Theory](#rag-fusion-theory)
- [Reciprocal Rank Fusion](#reciprocal-rank-fusion)
- [Memory Management](#memory-management)
- [Prompt Engineering](#prompt-engineering)
- [Project File Structure](#project-file-structure)
- [Environment Variables](#environment-variables)
- [Packages & Versions](#packages--versions)
- [Render Deployment](#render-deployment)
- [Routes](#routes)
- [How to Run Locally](#how-to-run-locally)

---

## 🌿 Overview

**VeganChatbotRAG** is a full-stack AI chatbot that answers questions exclusively from a curated vegan knowledge base (PDFs and text files). It leverages:

- **Pinecone** as the vector database for semantic search
- **BAAI/bge-small-en-v1.5** embeddings via HuggingFace
- **Groq (LLaMA 3.3 70B)** as the LLM backbone
- **RAG Fusion** to generate multiple queries from a single user question
- **Reciprocal Rank Fusion (RRF)** to rerank and merge results for highest accuracy
- **LangChain SQLite Memory** to persist multi-turn conversation history
- **Flask** for the backend web server
- **Render** for cloud deployment with auto-deploy via `render.yaml`

---

## 🏗️ Architecture

```
User Input (Flask UI)
        │
        ▼
 ┌─────────────────────────────┐
 │   RAG Fusion Query Generator│  ◄─── Groq LLM generates 3-5 sub-queries
 └────────────┬────────────────┘
              │ Multiple Queries
              ▼
 ┌─────────────────────────────┐
 │   Pinecone Vector Search     │  ◄─── BAAI/bge-small-en-v1.5 Embeddings
 └────────────┬────────────────┘
              │ Multiple Result Sets
              ▼
 ┌─────────────────────────────┐
 │  Reciprocal Rank Fusion      │  ◄─── Merge + Rerank all result lists
 └────────────┬────────────────┘
              │ Top-K Unified Results
              ▼
 ┌─────────────────────────────┐
 │  Prompt + Memory Assembly    │  ◄─── SQLite chat history injected
 └────────────┬────────────────┘
              │ Full Context Prompt
              ▼
 ┌─────────────────────────────┐
 │   Groq LLM (LLaMA 3.3 70B)  │  ◄─── Final answer generation
 └────────────┬────────────────┘
              │
              ▼
       Flask Response → UI
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 RAG Fusion | Expands a single user query into multiple sub-queries for broader retrieval |
| 🏆 Reciprocal Rank Fusion | Merges ranked lists from all sub-queries to find most relevant documents |
| 🧠 SQLite Memory | Full conversation history persisted per session using LangChain |
| 🌐 Pinecone Vector DB | Stores BAAI embeddings of vegan PDFs and text datasets |
| 🤖 Groq LLaMA 3.3 70B | Fast inference LLM for generating plant-based answers |
| 🌱 Vegan-Only Data | All answers grounded in curated vegan dataset |
| 🎛️ Memory UI Page | View, manage, and clear memory per session from the browser |
| ☁️ Render Auto-Deploy | Push to Git → automatically builds and deploys on Render |

---

## 🔬 RAG Fusion Theory

**Standard RAG** retrieves documents using a single query, which can miss relevant information if the query is phrased in a non-ideal way.

**RAG Fusion** solves this by:

1. **Multi-Query Generation** — The LLM generates 3–5 semantically diverse reformulations of the original question.
2. **Parallel Retrieval** — Each reformulated query is sent to Pinecone independently, producing multiple ranked result lists.
3. **Fusion** — All result lists are merged using Reciprocal Rank Fusion to produce a single unified ranked list.
4. **Answer Generation** — The top-K fused documents are passed into the final LLM prompt.

**Why it works:** Different phrasings retrieve different documents. Fusion ensures no relevant document is missed due to query-document vocabulary mismatch.

```
Original Query: "What are vegan protein sources?"
         │
         ▼
Sub-Query 1: "high protein plant-based foods"
Sub-Query 2: "best protein in vegan diet"
Sub-Query 3: "legumes seeds nuts protein content"
Sub-Query 4: "complete protein for vegans"
         │
         ▼
   4 × Pinecone searches → 4 ranked result lists
         │
         ▼
   RRF merges → 1 unified top-K result list
```

---

## 🏆 Reciprocal Rank Fusion

**RRF** is a ranking algorithm that combines multiple ranked lists without needing scores. Each document receives a score based on its rank in each list:

```
RRF_score(doc) = Σ  1 / (k + rank_i(doc))
                 i
```

Where:
- `k` is a constant (typically 60) that dampens the effect of high-ranked documents
- `rank_i(doc)` is the document's rank in the i-th result list

**Advantages:**
- No need to normalize or calibrate scores across different retrievers
- Robust to outlier rankings
- Consistently outperforms individual retrieval methods

**Example:**
```
List 1: [DocA(1), DocB(2), DocC(3)]
List 2: [DocC(1), DocA(2), DocD(3)]

RRF Score(DocA) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
RRF Score(DocC) = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
RRF Score(DocB) = 1/(60+2) = 0.01613

Final Order: DocA > DocC > DocB > DocD
```

---

## 🧠 Memory Management

Memory is implemented using **LangChain's `SQLChatMessageHistory`** backed by a local SQLite database (`chat_memory.db`).

```python
# src/memory.py
from langchain_community.chat_message_histories import SQLChatMessageHistory

DB_CONNECTION = "sqlite:///chat_memory.db"

def get_memory(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=DB_CONNECTION
    )
```

### How Memory Connects to RAG

1. On each user message, the session's full chat history is loaded from SQLite.
2. The history is formatted and injected into the prompt context (so the LLM knows prior turns).
3. After the LLM responds, both the user message and AI response are saved back to SQLite.
4. The RAG retrieval is performed on the **current user query** — not on the full history — keeping retrieval focused.

### Memory Routes

| Route | Method | Description |
|---|---|---|
| `/` | GET | Main chat interface |
| `/chat` | POST | Send message, get AI response with memory + RAG |
| `/memory` | GET | View full memory history for current session |
| `/clear-memory` | POST | Clear all memory for the current session |

### Session Management

- Each browser session gets a unique `session_id` stored in a cookie.
- The default session is `default_session` unless otherwise set.
- Memory is **per-session** — different users/browsers have isolated histories.

### Clearing Memory

From the **Memory History** page (shown in screenshot above), users can:
1. Click **"Clear Memory"** to wipe the current session's entire conversation history from SQLite.
2. The page shows total chats, all previous Q&A pairs, and timestamps.
3. After clearing, clicking **"Back to Chat"** returns to a fresh conversation.

---

## 💬 Prompt Engineering

The system uses structured prompt templates to:

1. **Inject RAG context** — retrieved vegan documents are formatted and inserted.
2. **Inject chat history** — previous turns create conversational continuity.
3. **Enforce vegan-only answers** — the system prompt instructs the LLM to only answer from provided context.
4. **Handle hallucination** — if the retrieved context doesn't contain an answer, the LLM is instructed to say "I don't have information on that from my vegan knowledge base" rather than hallucinating.

```python
# Example prompt structure (src/prompt.py)
system_prompt = """
You are VeganAI, a knowledgeable plant-based assistant.
Answer ONLY based on the provided context from vegan knowledge base.
If the answer is not in the context, say you don't know from your vegan data.
Do not hallucinate or invent information.

Context:
{context}

Chat History:
{chat_history}
"""
```

---

## 📁 Project File Structure

```
VeganChatbotRAG/
│
├── src/
│   ├── __init__.py            # Package init
│   ├── tools/                 # Utility tools
│   ├── helper.py              # Helper functions (text cleaning, formatting)
│   ├── llm_config.py          # Groq LLM configuration
│   ├── logger.py              # Logging setup
│   ├── memory.py              # SQLite chat memory (LangChain)
│   ├── prompt.py              # System & user prompt templates
│   ├── retrieval_fusion.py    # RAG Fusion + RRF implementation
│   └── store_index.py         # Pinecone index creation & embedding storage
│
├── static/
│   ├── style.css              # Main chat UI styles
│   └── memory.css             # Memory history page styles
│
├── templates/
│   ├── index.html             # Main chat UI (VeganAI chatbot)
│   └── memory.html            # Memory history & management page
│
├── research/                  # Jupyter notebooks / experimentation
│
├── app.py                     # Flask application entry point
├── index_data.py              # Script to index vegan PDFs/text into Pinecone
├── pdftext.py                 # PDF text extraction utility
├── setup.py                   # Package setup
├── chat_memory.db             # SQLite memory database (auto-created)
│
├── .env                       # Environment variables (never commit)
├── .gitignore
├── .python-version            # Python version pin (3.11.9)
├── requirements.txt           # All Python dependencies with versions
├── render.yaml                # Render cloud deployment config
└── README.md                  # This file
```

### Key File Explanations

**`app.py`** — Flask routes: `/` (chat UI), `/chat` (POST API), `/memory` (history view), `/clear-memory` (POST to clear SQLite session).

**`src/retrieval_fusion.py`** — Core RAG logic: generates sub-queries via Groq, runs parallel Pinecone searches, applies RRF, returns top-K merged documents.

**`src/memory.py`** — Returns a `SQLChatMessageHistory` instance for a given session, connected to `chat_memory.db`.

**`src/llm_config.py`** — Initializes the Groq `ChatGroq` client with `llama-3.3-70b-versatile`.

**`src/prompt.py`** — Defines LangChain `ChatPromptTemplate` with system instructions, context placeholder, and chat history.

**`index_data.py`** — Run once to process all vegan PDFs/text files, embed them using `BAAI/bge-small-en-v1.5`, and upsert into Pinecone.

**`src/store_index.py`** — Manages Pinecone index creation and the `HuggingFaceEmbeddings` setup.

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

> ⚠️ **Never commit `.env` to Git.** It is listed in `.gitignore`.

---

## 📦 Packages & Versions

```
Flask==3.1.3
gunicorn==21.2.0
python-dotenv==1.0.1

# LangChain ecosystem
langchain==0.2.16
langchain-core==0.2.38
langchain-community==0.2.16
langchain-pinecone==0.1.2
langchain-groq==0.1.9

# Vector Database
pinecone==3.2.2

# Embeddings
sentence-transformers==2.7.0
huggingface-hub==0.23.4
torch==2.2.2

# PDF Processing
pypdf==4.2.0

# Groq LLM
groq==0.9.0

# Database / HTTP
SQLAlchemy>=2.0.0
httpx==0.27.0
```

### Embedding Model

```python
model_name = "BAAI/bge-small-en-v1.5"
```

`BAAI/bge-small-en-v1.5` is a high-quality, lightweight English embedding model from Beijing Academy of AI:
- **Dimension:** 384
- **Optimized for:** semantic similarity and retrieval tasks
- **Why chosen:** excellent accuracy-to-speed ratio; runs efficiently even on CPU

---

## ☁️ Render Deployment

The project auto-deploys to [Render](https://render.com) using `render.yaml`:

```yaml
services:
  - type: web
    name: veganrag-chatbot
    runtime: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
    plan: free
    autoDeploy: true

    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: HUGGINGFACEHUB_API_TOKEN
        sync: false
      - key: PINECONE_API_KEY
        sync: false
      - key: GROQ_API_KEY
        sync: false
      - key: GROQ_MODEL
        value: llama-3.3-70b-versatile
```

### Deployment Steps

1. Push the project to a GitHub repository.
2. Connect the repo to Render.
3. Set secret environment variables (`HUGGINGFACEHUB_API_TOKEN`, `PINECONE_API_KEY`, `GROQ_API_KEY`) in the Render dashboard under **Environment**.
4. Every `git push` to `main` triggers an automatic rebuild and redeploy.

> **Note:** The SQLite `chat_memory.db` is ephemeral on Render's free tier — it resets on each deploy. For production persistence, migrate to a hosted PostgreSQL database with LangChain's `PostgresChatMessageHistory`.

---

## 🚀 How to Run Locally

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/VeganChatbotRAG.git
cd VeganChatbotRAG
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Index Your Vegan Data

Place all vegan PDFs and text files in a `data/` folder, then:

```bash
python index_data.py
```

This embeds and upserts all documents into Pinecone. Run **once** (or whenever you add new data).

### 4. Run the Flask App

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 🔄 Application Routes

| Route | Method | Page/Action |
|---|---|---|
| `/` | `GET` | Main VeganAI chat interface |
| `/chat` | `POST` | Accepts `{"message": "...", "session_id": "..."}`, returns AI response |
| `/memory` | `GET` | Memory History page — view all past Q&A for this session |
| `/clear-memory` | `POST` | Clears all chat history for the session in SQLite |

### Memory Flow

```
User visits /          ← Flask sets session cookie (session_id)
User sends message     ← POST /chat
  └─ Load memory from SQLite (session_id)
  └─ RAG Fusion retrieval
  └─ RRF reranking
  └─ Build prompt (context + history)
  └─ Groq LLM generates response
  └─ Save user + AI messages to SQLite
  └─ Return response to UI

User visits /memory    ← Load & display all messages from SQLite
User clicks Clear      ← POST /clear-memory → wipe SQLite for session_id
User clicks Back       ← Redirect to /
```

---

## 🌱 About the Vegan Dataset

The knowledge base consists of curated vegan content including:
- Nutrition guides and protein source PDFs
- Plant-based recipe collections
- Vegan lifestyle and ethics documents
- Research papers on plant-based diets

All data is indexed once via `index_data.py` and stored in Pinecone. The chatbot **only answers from this data** — it will not hallucinate information outside the vegan knowledge base.

---

## 🙏 Tech Stack Summary

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B Versatile |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Vector DB | Pinecone |
| RAG Strategy | RAG Fusion + Reciprocal Rank Fusion |
| Memory | LangChain SQLChatMessageHistory (SQLite) |
| Framework | Flask |
| Deployment | Render (render.yaml, auto-deploy) |
| PDF Parsing | PyPDF |
| Orchestration | LangChain |

---

*Made with 🌿 for the plant-based community*
