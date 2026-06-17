from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


data = PyPDFLoader("documents loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap= 200
)

chunk = splitter.split_documents(docs)

template= ChatPromptTemplate.from_messages([
    ("system", "you are a Ai that summaries the text"),
    ("human","{data}")
])

model = ChatMistralAI(model="mistral-small-2603")
prompt = template.format_messages(data= docs)

response = model.invoke(prompt)

print(response.content)