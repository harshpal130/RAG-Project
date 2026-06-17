from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
from langchain_core.documents import Document

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

docs = [
    Document(page_content="python is widly used in Ai.", metadata={"souce:": "1eklvn"}),
    Document(page_content="pandas are used for data analysis in python.", metadata={"souce:": "2fdeklvn"}),
    Document(page_content="neural network are used in deep learning.", metadata={"souce:": "3lvn"})
]

vectorStore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorStore.similarity_search("what is used for data analysis" , k=2)

for r in result:
    print(r.page_content)

retriver = vectorStore.as_retriever()

docs = retriver.invoke()