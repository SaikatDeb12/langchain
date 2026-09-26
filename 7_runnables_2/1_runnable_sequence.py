from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()
llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="tell me a joke about the topic : {topic}", input_variables=["topic"]
)

parser = StrOutputParser()


prompt2 = PromptTemplate(
    template="explain me the this joke : \n {joke}", input_variables=["joke"]
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

response = chain.invoke(
    {
        "topic": "computer",
    }
)
print(response)

# response :
# Why did the computer show up late to work?
#
# Because it had a **hard drive**!
