from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="generate a 1-2 lines tweet about the topic : {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a 2-3 lines linkedIn post about the topic : {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

paralledChain = RunnableParallel(
    {
        "twitter": prompt1 | model | parser,
        "linkedIn": prompt2 | model | parser,
    }
)

response = paralledChain.invoke({"topic": "AI"})

print(response["twitter"])
print("---------------------")
print(response["linkedIn"])
