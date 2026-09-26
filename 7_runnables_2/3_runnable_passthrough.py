from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", temperature=1.5)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="give me a joke about the topic : {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="give me a short expaination on the joke : \n {joke}",
    input_variables=["joke"],
)

jokeGenerateChain = RunnableSequence(prompt1, model, parser)

parallelChain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "meaning": RunnableSequence(prompt2, model, parser),
    }
)


finalChain = RunnableSequence(jokeGenerateChain, parallelChain)
response = finalChain.invoke({"topic": "instagram"})

print(response)
