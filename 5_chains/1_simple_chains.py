from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", task="text-generation")

prompt = PromptTemplate(
    template="Generate 5 interesting facts about the topic : {topic}",
    input_variables=["topic"],
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser
response = chain.invoke({"topic": "universe"})

print(response)
