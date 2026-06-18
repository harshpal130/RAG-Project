# 📚 BookMind AI

BookMind AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and chat with them using natural language. The application extracts content from PDFs, converts it into vector embeddings, stores them in ChromaDB, and uses Mistral AI to generate context-aware responses.

🌐 **Live Demo:** https://bookmind-ai.streamlit.app/

---

## 🚀 Features

- 📄 Upload and chat with any PDF document
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic search using vector embeddings
- 💬 Interactive chat interface
- 🔄 Replace uploaded PDFs instantly
- 🗑️ Delete current document and start fresh
- 🧹 Clear chat history
- 📊 View PDF statistics (pages and chunks)
- ⚡ Fast retrieval with ChromaDB
- 🎨 Clean and responsive Streamlit UI

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend
- Python
- LangChain

### Vector Database
- ChromaDB

### Embedding Model
- sentence-transformers/all-MiniLM-L6-v2

### LLM
- Mistral AI (mistral-small-2506)

### Document Processing
- PyPDFLoader
- RecursiveCharacterTextSplitter

---

## 📂 Project Structure

```text
BookMind-AI/
│
├── app.py
├── requirements.txt
├── .env
├── chroma_db/
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/bookmind-ai.git
cd bookmind-ai
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

Get your API key from:

https://console.mistral.ai/

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🧠 How It Works

1. User uploads a PDF document.
2. PDF content is extracted using PyPDFLoader.
3. Text is split into smaller chunks.
4. Chunks are converted into embeddings using HuggingFace sentence transformers.
5. Embeddings are stored in ChromaDB.
6. Relevant chunks are retrieved using MMR retrieval.
7. Mistral AI generates answers based only on retrieved context.

---

## 💡 Example Questions

```text
What is the main topic of this document?

Summarize chapter 2.

Explain the key concepts discussed.

What conclusions are mentioned in the paper?

Give me a brief overview of the document.
```

---

## 🔮 Future Enhancements

- Multiple PDF support
- Source citations with page numbers
- Streaming responses
- Download chat history
- Dark mode support
- User authentication
- Cloud vector database integration

---

## 🌐 Live Demo

https://bookmind-ai.streamlit.app/

---

## 👨‍💻 Author

Developed with ❤️ using LangChain, ChromaDB, Mistral AI, and Streamlit.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.
