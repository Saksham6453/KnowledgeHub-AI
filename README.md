# 📚 KnowledgeHub AI

> Chat with **PDFs, DOCX, TXT files, Websites, and YouTube Videos** using **Retrieval-Augmented Generation (RAG)** powered by **LangChain, NVIDIA AI Endpoints, FAISS, and Streamlit.**

---

## 🚀 Demo

**Live Demo:** https://knowledgeapp-ai-qdesbseoqi5mhb2srqgpva.streamlit.app/

---

## ✨ Features

- 📄 Chat with PDF documents
- 📝 Chat with TXT files
- 📃 Chat with DOCX documents
- 🌐 Chat with Websites
- 🎥 Chat with YouTube videos
- 🔍 Semantic Search using FAISS
- 🧠 NVIDIA Nemotron Embeddings
- 🤖 NVIDIA Llama 3.1 8B Instruct
- 💬 Conversational RAG Pipeline
- ⏱️ YouTube timestamp navigation
- 📚 Retrieved Sources Display
- 🎨 Modern Streamlit UI

---

# 🏗️ Architecture

```text
                 User Uploads
                       │
        ┌──────────────┼──────────────┐
        │              │              │
      PDF           Website       YouTube
        │              │              │
        └──────────────┼──────────────┘
                       │
               Document Loaders
                       │
                       ▼
              Recursive Text Splitter
                       │
                       ▼
           NVIDIA Nemotron Embeddings
                       │
                       ▼
                 FAISS Vector Store
                       │
                       ▼
                  Retriever (Top-3)
                       │
                User Question
                       │
                       ▼
              Similarity Search
                       │
                       ▼
               Retrieved Context
                       │
                       ▼
            NVIDIA Llama 3.1 8B
                       │
                       ▼
                 Final Response
```

---

# 🛠️ Tech Stack

| Technology | Usage |
|------------|------|
| Python | Backend |
| Streamlit | Frontend |
| LangChain | RAG Framework |
| NVIDIA AI Endpoints | LLM & Embeddings |
| FAISS | Vector Database |
| PyPDF | PDF Loader |
| Docx2txt | DOCX Loader |
| BeautifulSoup | Website Loader |
| YouTube Transcript API | YouTube Transcript Extraction |

---

# 📂 Project Structure

```text
KnowledgeHub-AI/
│
|-- assets/
│── home_page.png
│── Pdf_page.png
│── Website PAGE.png
│── YT_page.png
│
├── app.py                 # Streamlit Frontend
├── rag.py                 # Complete RAG Pipeline
├── requirements.txt
├── .env
├── README.md
│
└── .venv/



# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/KnowledgeHub-AI.git

cd KnowledgeHub-AI
```

Create Virtual Environment

```bash
python -m venv .venv
```

Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file inside the project root.

```env
NVIDIA_API_KEY=YOUR_API_KEY
```

Get your free NVIDIA API key from:

https://build.nvidia.com/

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📥 Supported Knowledge Sources

| Source | Supported |
|---------|-----------|
| PDF | ✅ |
| DOCX | ✅ |
| TXT | ✅ |
| Website | ✅ |
| YouTube | ✅ |

---

# 🔄 RAG Workflow

### Indexing Phase

- Load Documents
- Split Documents
- Generate Embeddings
- Create FAISS Vector Store
- Build Retriever

---

### Query Phase

- User asks a question
- Retrieve Top-K relevant chunks
- Create Context
- Prompt LLM
- Generate Answer

---

# 📦 Requirements

- Python 3.10+
- NVIDIA API Key
- Internet Connection

---

# 💡 Future Improvements

- Conversation Memory
- Chat History Persistence
- ChromaDB Support
- Pinecone Support
- Hybrid Search
- OCR Support
- Image Question Answering
- Multi-PDF Chat
- Citation Highlighting
- Authentication
- Docker Deployment

---

# 🎯 Learning Outcomes

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- LangChain Pipelines
- Prompt Engineering
- FAISS Vector Search
- Semantic Search
- NVIDIA AI Endpoints
- Document Processing
- Streamlit Development
- Multi-Source Knowledge Retrieval



# 👨‍💻 Author

**Saksham Sharma**
**Ai Engineer**

- GitHub: https://github.com/Saksham6453

---

## ⭐ If you found this project helpful, consider giving it a Star!
