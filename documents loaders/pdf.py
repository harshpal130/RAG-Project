from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader("deeplearning.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000000,
    chunk_overlap= 10,
)

chunk = splitter.split_documents(docs)

print(chunk[0].page_content)