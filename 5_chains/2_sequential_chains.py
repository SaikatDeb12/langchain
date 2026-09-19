from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Give me a detailed report on the topic : {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Give me a summary of the text : \n {text}", input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser
response = chain.invoke(
    {
        "topic": "linear regression",
    }
)

print(response)
chain.get_graph().print_ascii()
