# even if we store messages in a chat, we need to have proper record of the messages from user and AI respectively
# so instead of maintaining a list we need to have a dictionary

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

# two types of messages system, human, AI message
# system - the initial message / information we send to AI (eg : "you are a helpful assistant")
# human - it is the user input
# AI - it is the response from the AI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

chatHistory = [
    SystemMessage(content="you are a helpul assistant"),
    HumanMessage(content="what is langchain"),
]

response = model.invoke(chatHistory)
chatHistory.append(AIMessage(content=response.content[0]["text"]))
print(chatHistory)
