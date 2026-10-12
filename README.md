# 🛍️ ShopMate — AI E-Commerce Shopping Assistant

A full-stack, production-ready AI shopping assistant that combines natural language understanding, semantic product search over local vector embeddings, order status tracking via a dedicated FastAPI microservice, conversational follow-up memory, and a modern Streamlit interface.

---

## 🌟 Key Features

- **Semantic Product Discovery:** Understands user intent and natural language shopping queries (e.g., *"Find wireless headphones for studying"* or *"Show me a gift for a student"*) using Sentence Transformers (`all-MiniLM-L6-v2`) and a persistent ChromaDB vector store.
- **Strict Budget & Category Filtering:** Accurately extracts numeric budgets (e.g., *"under ₹3,000"*, *"below 2500 rs"*) and enforces hard filtering in code so products over budget are never recommended.
- **Live Order Status Tracking:** Direct integration with a FastAPI microservice providing order statuses across 6 distinct phases (*Processing*, *Confirmed*, *Packed*, *Shipped*, *Out for delivery*, *Delivered*).
- **Graceful Fallback Mode:** Operates autonomously with zero external service dependencies. If the FastAPI server is offline, it automatically switches to local mock fallback data with clear visual indicators.
- **Dual-Mode AI Reasoning:**
  - **Mode A (LLM Enabled):** Uses OpenAI or any compatible API provider (e.g. Groq, Ollama, vLLM) when `OPENAI_API_KEY` is provided.
  - **Mode B (Zero-Cost Deterministic Fallback):** Runs completely offline without API keys, using algorithmic ranking, specification comparison, and structured responses.
- **Contextual Conversation Memory:** Tracks active budgets, preferred categories, and recently recommended products across turns to answer comparative queries like *"Which one has the highest rating?"* or *"Show me cheaper alternatives"*.
- **Modern E-Commerce UI:** Polished, responsive Streamlit dashboard with custom CSS, structured product cards, real-time status badges, currency formatting in Indian Rupees (₹), and quick-action prompts.

---

## 🏗️ Architecture Overview

```
                                  ┌───────────────────────────┐
                                  │    User Browser / Chat    │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Streamlit UI (app.py)   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   ShoppingAssistant       │
                                  │      (assistant.py)       │
                                  └──────┬────────────┬───────┘
                                         │            │
                   ┌─────────────────────┘            └─────────────────────┐
                   ▼                                                        ▼
    ┌─────────────────────────────┐                          ┌─────────────────────────────┐
    │  Intent Router & Extraction │                          │    Conversation Memory      │
    │      (intent_router.py)     │                          │        (memory.py)          │
    └──────────────┬──────────────┘                          └─────────────────────────────┘
                   │
         ┌─────────┴──────────────────────┐
         ▼                                ▼
┌──────────────────┐            ┌──────────────────┐
│  Product Search  │            │   Order Client   │
│(product_search.py│            │ (order_client.py)│
└────────┬─────────┘            └────────┬─────────┘
         │                               │
         ▼                               ▼ HTTP / Fallback
┌──────────────────┐            ┌─────────────────────────────┐
│ ChromaDB Vector  │            │ FastAPI Order Microservice  │
│ Index + CSV Data │            │     (order_service.py)      │
└──────────────────┘            └─────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Language** | Python 3.10+ | Core application runtime |
| **Frontend UI** | Streamlit | Responsive conversational dashboard |
| **Backend API** | FastAPI + Uvicorn | High-performance mock order service |
| **Vector Database** | ChromaDB | Local persistent vector storage |
| **Embeddings** | Sentence Transformers | `all-MiniLM-L6-v2` (384-dimensional dense vectors) |
| **Data Processing** | Pandas | CSV loading, type casting, validation |
| **HTTP Client** | HTTPX | Asynchronous/synchronous REST client with timeouts |
| **Testing** | Pytest | 23 automated unit and integration tests |
| **Configuration** | python-dotenv | Environment variable management |

---

## 📁 Project Structure

```
shopmate/
├── app.py                     # Streamlit web application & UI components
├── assistant.py               # AI coordinator (routing, retrieval, generation)
├── product_search.py          # ChromaDB semantic search & CSV validation
├── order_service.py           # FastAPI mock order status microservice
├── order_client.py            # HTTP client for order API with fallback
├── orders_data.py             # Centralized mock orders dataset
├── intent_router.py           # Natural language intent & entity parsing
├── memory.py                  # Conversation history and entity memory
├── requirements.txt           # Pinned dependency requirements
├── .env.example               # Template for environment configuration
├── .gitignore                 # Version control exclusions
├── README.md                  # Comprehensive documentation
├── data/
│   └── products.csv           # 24 realistic sample products catalog
├── tests/
│   ├── __init__.py
│   ├── test_orders.py         # Tests for FastAPI & OrderClient
│   ├── test_product_search.py # Tests for ChromaDB, CSV, and search
│   ├── test_intent_router.py  # Tests for routing and budget parsing
│   └── test_memory.py         # Tests for conversation turn memory
└── .streamlit/
    └── config.toml            # Streamlit theme & server configuration
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Git (optional)

### 2. Virtual Environment Setup

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` (optional):
```env
ORDER_API_URL=http://127.0.0.1:8000

# Optional: To enable OpenAI completions, supply your API key:
OPENAI_API_KEY=
OPENAI_MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=
```
> **Note:** An API key is strictly optional! If `OPENAI_API_KEY` is not provided, ShopMate automatically runs in its deterministic offline mode with zero cost.

---

## 🖥️ Running the Application

For full end-to-end functionality with the separate microservice, run the following two commands:

### Terminal 1: Launch FastAPI Order Service
```bash
uvicorn order_service:app --reload --port 8000
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Terminal 2: Launch Streamlit Frontend
```bash
streamlit run app.py
```
- Open your browser at: `http://localhost:8501`

---

## 💬 Sample Prompts & Interactions

### 1. Natural Language Product Search with Budget Limits
- *"Find wireless headphones under ₹3000"*
- *"I need a laptop accessory for studying"*
- *"Show me a speaker under ₹2500"*
- *"Show mechanical keyboards"*

### 2. Multi-Turn Context & Comparisons
- **User:** *"I need wireless headphones under ₹3000"*
- **Assistant:** Recommends matching items (e.g. boAt Rockerz 450, OnePlus Bullets Z2).
- **User:** *"Which one has the highest rating?"*
- **Assistant:** Analyzes previous recommendations and identifies the top-rated item.
- **User:** *"Show me cheaper alternatives"*
- **Assistant:** Automatically searches lower-priced audio options within the established context.

### 3. Live Order Tracking
- *"Where is my order ORD1002?"*
- *"Track order ORD1003"*
- *"What is the delivery status of ORD1001?"*

### 📋 Demo Order Reference

| Order ID | Product | Status | Carrier |
|---|---|---|---|
| `ORD1001` | boAt Rockerz 450 Headphones | **Delivered** | BlueDart Express |
| `ORD1002` | Sony WH-1000XM4 Headphones | **Shipped** | Delhivery Priority |
| `ORD1003` | Keychron K2 V2 Keyboard | **Out for delivery** | Ekart Logistics |
| `ORD1004` | Apple iPad 10th Gen | **Packed** | BlueDart Express |
| `ORD1005` | Portronics Toad 23 Mouse | **Confirmed** | Shadowfax Logistics |
| `ORD1006` | JBL Flip 6 Speaker | **Processing** | FedEx India |
| `ORD1007` | Noise ColorFit Pro 4 Watch | **Delivered** | Delhivery Priority |

---

## 🧪 Running Automated Tests

Run the complete test suite with Pytest:

```bash
pytest -v
```

The test suite contains 23 tests verifying:
- FastAPI health check and order endpoints
- Case-insensitive order ID resolution and 404 handling
- OrderClient HTTP requests and offline fallback behavior
- CSV catalog loading, validation, and missing column handling
- ChromaDB vector indexing and persistence in isolated temporary directories
- Strict numeric price constraints (asserting 0 products exceed max budget)
- Category filtering and empty result handling
- Intent classification and budget entity parsing
- Conversation memory windowing and entity state tracking

---

## ☁️ Deployment Guide

### Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and create a new app pointing to your repository with `app.py` as the main entry point.
3. Configure **Secrets** in the Streamlit Cloud Dashboard:
   ```toml
   # In Streamlit Cloud -> App Settings -> Secrets:
   ORDER_API_URL = "https://your-hosted-order-api.com"  # (Optional: point to external FastAPI deployment)
   OPENAI_API_KEY = "sk-..."                            # (Optional: if enabling LLM)
   ```

### ⚠️ Important Hosting Notice regarding `127.0.0.1`

When hosted on Streamlit Community Cloud:
- `127.0.0.1` or `localhost` refers to the container hosting the Streamlit app, **not** your personal computer or a separate local process.
- Streamlit Community Cloud runs `streamlit run app.py` as a single process.
- **ShopMate handles this out-of-the-box:**
  - If a separately hosted FastAPI URL is provided in `ORDER_API_URL`, ShopMate connects to it.
  - If no separate API is running, ShopMate **automatically activates its local mock fallback**, ensuring all demo orders (`ORD1001`–`ORD1007`) continue to work seamlessly for evaluators and users!
- ChromaDB automatically regenerates its vector embeddings from `data/products.csv` if the persistent vector cache is cleared during container rebuilds.

---

## 🔒 Grounding & Safety Guarantees

- **No Hallucinated Products:** The assistant only returns products that exist in `data/products.csv`.
- **No Hallucinated Order Tracking:** Shipment checkpoints, delivery dates, and tracking codes are strictly retrieved from real API data or verified mock records.
- **Zero Secret Leakage:** Stack traces, internal paths, and API keys are strictly kept server-side and never exposed in chat output.
