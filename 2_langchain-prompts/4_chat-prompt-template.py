from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

chatTemplate = ChatPromptTemplate(
    [
        ("system", "You are a helpful {domain} expert"),
        ("human", "Explain me in short, what is {topic}"),
    ]
)

# chatTemplate = [
# SystemMessage(content="You are a helpful {domain} expert"),
# HumanMessage(content="Explain me in short, what is {topic}"),
# ]

prompt = chatTemplate.invoke(
    {"domain": "cyber security expert", "topic": "capture the flag"}
)
print(prompt)
