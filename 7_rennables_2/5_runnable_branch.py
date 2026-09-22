from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
    RunnableSequence,
)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", temperature=1.5)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="give me a detailed report on the topic : \n {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="give me a short summary on the topic : \n {topic}",
    input_variables=["topic"],
)


def wordCount(text):
    return len(text.split())


generateReportChain = RunnableSequence(prompt, model, parser)

branchChain = RunnableBranch(
    (lambda x: len(x.split()) > 150, RunnableSequence(prompt2, model, parser)),
    (RunnablePassthrough()),
)

# LCEL
finalChain = generateReportChain | branchChain

response = finalChain.invoke({"topic": "capture the flag (ctfs)"})

print(response)
