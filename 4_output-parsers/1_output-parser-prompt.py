from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it",
    task="text-generation",
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)

# prompt 1
template1 = PromptTemplate(
    template="You need to provide a short 10 lines report on the topic : {topic}",
    input_variables=["topic"],
)

# prompt 2
template2 = PromptTemplate(
    template="Give a short summary on the following line: \n {text}",
    input_variables=["text"],
)

# here we can also use template1.format()
prompt1 = template1.invoke({"topic": "black hole"})
print(prompt1)
response1 = model.invoke(prompt1)
print(response1.content)

prompt2 = template2.invoke({"text": response1.content})
print(prompt2)
response2 = model.invoke(prompt2)

print(response2.content)
