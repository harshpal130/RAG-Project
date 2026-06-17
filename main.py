from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 

vectorStore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriver = vectorStore.as_retriever(
    search_type ="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5,  #0 for unrelated answer and 1 for releated answer

    }
)

llm = ChatMistralAI(model="mistral-small-2506")

# prompt template

promt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are helpful AI agent ..
         use ONLY provided context  to answer the question.
         if answer is not present in the context,
         say:I could not find the answer in the document 
         """),
         ("human",
          
          """Context:{context}.
          Question :{question}.
          """
          
          )
    ]
)

print("rag system created ")

print("press 0 to exit ")

while True:
    query = input("You:")
    if query == "0":
        break

    docs = retriver.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )  

    final_prompt = promt.invoke({
        "context":context,
        "question":query
    })  

    response = llm.invoke(final_prompt)
    print(f"\n AI: {response.content}")