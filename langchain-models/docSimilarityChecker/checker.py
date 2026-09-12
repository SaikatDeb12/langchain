from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddingModel = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

documents = [
    "Python is a popular programming language used for web development and data science.",
    "Machine learning allows computers to learn patterns from data without explicit programming.",
    "JavaScript is widely used for building interactive web applications.",
    "Deep learning is a subset of machine learning that uses neural networks.",
    "FastAPI is a Python framework for building high-performance web APIs.",
]

query = "want to develop a web application which programming language should i use"
# query = "want to develop a interactive web application which programming language should i use"

docEmbeddings = embeddingModel.embed_documents(documents)
queryEmbeddings = embeddingModel.embed_query(query)

# print(cosine_similarity([queryEmbeddings], docEmbeddings))
scores = cosine_similarity([queryEmbeddings], docEmbeddings)[
    0
]  # first list in a 2d list

# print(sorted(list(enumerate(scores)), key=lambda x: x[1])[-1])
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(documents[index])
