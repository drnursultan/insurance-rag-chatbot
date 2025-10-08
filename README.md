# 🧠 Insurance RAG Chatbot

An intelligent **Retrieval-Augmented Generation (RAG)** chatbot that answers insurance-related questions using **local documents** as ground truth.  
Built with **LangChain**, **ChromaDB**, and **HuggingFace models**, this project demonstrates how enterprise chatbots (like those in insurance or banking) can deliver accurate, grounded responses.

---

## 🚀 Overview

**Goal:**  
Create a local, explainable chatbot that can read insurance policies, FAQs, or billing documents and answer user questions based on context — not memorization.

**Key Features:**
- Retrieval-Augmented Generation (RAG) pipeline  
- Uses `sentence-transformers/all-MiniLM-L6-v2` for semantic search  
- Lightweight local model (`zephyr-7b-alpha`) for response generation  
- Interactive chat via terminal (`chat_cli.py`)  
- Automatic document loading and chunk embedding  
- Citations from the original document sources  

---

## 🧩 Project Structure

```
insurance-rag-chatbot/
├── data/                      # Text files used for training context
│   ├── insurance_manual.txt
│   ├── coverage_explained.txt
│   ├── claims_guide.txt
│   └── billing_support.txt
│
├── chroma/                    # Vector database storage
├── src/
│   ├── embed_store.py         # Builds embeddings & vector DB
│   ├── llm_providers.py       # Loads OpenAI or HF models
│   ├── config.py              # Model & directory configuration
│   └── rag_chain.py           # Core RAG chain logic
│
├── chat_cli.py                # Command-line chat interface
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone & install dependencies
```bash
git clone https://github.com/yourusername/insurance-rag-chatbot.git
cd insurance-rag-chatbot
pip install -r requirements.txt

### 2️⃣ Add your insurance text files

```bash
Place .txt documents inside the data/ folder.
You can include FAQs, policy manuals, or billing guides.

### 3️⃣ Build embeddings
```bash
python -m src.embed_store

### 4️⃣ Start the chatbot
```bash
python chat_cli.py

### 💬 Example Interaction
```
You: What does my coverage include?
Bot: Insurance protects against pooled risks, covering property damage,
theft, or liability. Dwelling and personal property are included.
(Source: coverage_explained.txt)
```

```
You: How can I contact billing support?
Bot: You can reach our billing department via the 24/7 hotline or
submit a ticket through your online account portal.
(Source: billing_support.txt)
```

## 🧠 Technologies Used
- **LangChain** – RAG orchestration  
- **ChromaDB** – local vector database  
- **HuggingFace Transformers** – embeddings & LLM pipeline  
- **PyTorch / MPS** – model inference acceleration  
- **SentenceTransformers** – text embeddings (`all-MiniLM-L6-v2`)

---

## 📊 Next Steps
- Add a **Streamlit UI** for web-based chat  
- Integrate **intent classifier** (billing / claims / coverage)  
- Connect to **external document APIs**  
- Add **evaluation metrics** (retrieval accuracy, answer grounding)

---

## 👨‍💻 Author
**Nursultan Azhimuratov** ([GitHub @drnursultan](https://github.com/drnursultan))  
PhD Candidate in Statistics · AI & NLP Developer  
If you use this repo, a ⭐️ or fork is appreciated!

---

## 📜 License
MIT License © 2025 Nursultan Azhimuratov