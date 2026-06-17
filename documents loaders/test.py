from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter= CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap = 1,
)

data = TextLoader("notes.txt")  # give output in form of object for that we need to unpack this use document loader

docs = data.load()

chunk = splitter.split_documents(docs)

for i in chunk:
    print()
    print(i.page_content)
    print()
    