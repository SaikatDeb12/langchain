from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
# vector = embeddings.embed_query("what is the meaning of life")
# print(vector[:5])

documents = [
    "hello world",
    "How are you",
    "what is your name?",
]

vectorDocs = embeddings.embed_documents(documents)
print(vectorDocs)
