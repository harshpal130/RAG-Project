import os
import uuid
import tempfile
import streamlit as st

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# ====================================================
# CONFIG
# ====================================================

load_dotenv()

st.set_page_config(
    page_title="BookMind AI",
    page_icon="📚",
    layout="wide"
)

# ====================================================
# CUSTOM CSS
# ====================================================

st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#4F8BF9;
}
.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>📚 BookMind AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload any PDF and chat with it instantly</div>",
    unsafe_allow_html=True
)

# ====================================================
# SESSION STATE
# ====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

if "db_path" not in st.session_state:
    st.session_state.db_path = None

# ====================================================
# MODELS
# ====================================================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embeddings()

llm = ChatMistralAI(
    model="mistral-small-2506"
)

# ====================================================
# PROMPT
# ====================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are BookMind AI.

            Answer ONLY from the provided context.

            If the answer is unavailable in the context say:

            "I could not find the answer in the document."
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)

# ====================================================
# FUNCTIONS
# ====================================================

def delete_database():
    st.session_state.retriever = None
    st.session_state.current_pdf = None
    st.session_state.messages = []
    st.session_state.pages = 0
    st.session_state.chunks = 0
    st.session_state.db_path = None


def process_pdf(uploaded_file):

    # Reset previous session
    delete_database()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Unique database for every upload
    db_path = f"chroma_db/{uuid.uuid4()}"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    st.session_state.retriever = retriever
    st.session_state.current_pdf = uploaded_file.name
    st.session_state.pages = len(docs)
    st.session_state.chunks = len(chunks)
    st.session_state.db_path = db_path
    st.session_state.messages = []

# ====================================================
# SIDEBAR
# ====================================================

with st.sidebar:

    st.header("⚙️ Control Panel")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    col1, col2 = st.columns(2)

    with col1:
        upload_btn = st.button(
            "📥 Upload / Replace PDF",
            use_container_width=True
        )

    with col2:
        delete_btn = st.button(
            "🗑 Delete",
            use_container_width=True
        )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    if st.session_state.current_pdf:

        st.success(
            f"Current PDF:\n\n{st.session_state.current_pdf}"
        )

        st.metric(
            "Pages",
            st.session_state.pages
        )

        st.metric(
            "Chunks",
            st.session_state.chunks
        )

# ====================================================
# DELETE CURRENT PDF
# ====================================================

if delete_btn:

    delete_database()

    st.success(
        "Current PDF removed successfully."
    )

    st.rerun()

# ====================================================
# INDEX PDF
# ====================================================

if upload_btn:

    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    else:

        with st.spinner(
            "Reading PDF and building vector database..."
        ):

            process_pdf(uploaded_file)

        st.success(
            f"{uploaded_file.name} indexed successfully!"
        )

        st.rerun()

# ====================================================
# DISPLAY CHAT
# ====================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====================================================
# CHAT INPUT
# ====================================================

if st.session_state.retriever:

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        docs = st.session_state.retriever.invoke(
            question
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                response = llm.invoke(
                    final_prompt
                )

                st.markdown(
                    response.content
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

else:

    st.info(
        """
        👈 Upload a PDF and click **Upload / Replace PDF**
        to start chatting.
        """
    )

# ====================================================
# FOOTER
# ====================================================

st.divider()

st.caption(
    "🚀 BookMind AI • Powered by LangChain + ChromaDB + Mistral AI"
)