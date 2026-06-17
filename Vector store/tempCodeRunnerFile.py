from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="python is widly used in Ai.", metadata={"souce:": "dlldeklvn"}),
    Document(page_content="java is widly used in games.", metadata={"souce:": "brafdeklvn"}),
    Document(page_content="c is widly used in bignner.", metadata={"souce:": "dlldekgergelvn"})
]

vectorStore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)