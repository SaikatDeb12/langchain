from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

chatTemplate = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support agent"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)

chatHistory = []

with open("chat_history.txt", "r") as f:
    chatHistory.extend(f.readlines())

print(chatHistory)

# creating the final prompt
prompt = chatTemplate.invoke(
    {
        "chat_history": chatHistory,
        "query": "where is my refund?",
    }
)

print(prompt)
