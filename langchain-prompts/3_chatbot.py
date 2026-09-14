from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

chatHistory = []

while True:
    userInput = input("You : ")
    if userInput == "exit":
        break
    chatHistory.append(HumanMessage(content=userInput))
    response = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=response.content[0]["text"]))
    print("AI: ", response.content[0]["text"])

print(chatHistory)
