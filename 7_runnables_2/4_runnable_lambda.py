from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Runnable lambda is a runnable primitive that helps us apply a python function within an AI pipline
# It act as a middleware between different components, enabling preprocessing, filtering, API calls

load_dotenv()
llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", temperature=1.5)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="write a joke about topic : {topic}", input_variables=["topic"]
)


def wordCount(text):
    return len(text.split(" "))


jokeGenerateChain = RunnableSequence(prompt, model, parser)

parallelChain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(wordCount)}
)

finalChain = RunnableSequence(jokeGenerateChain, parallelChain)
response = finalChain.invoke({"topic": "human"})

print(response)
