from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
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

parser = StrOutputParser()

# output parsers are used with chains
# template1(prompt) -> model -> response(output parsers) -> again,
# template2(prompt) -> model -> response(output parser)

# chain = (
#     template1
#     | model
#     | parser
#     | RunnableLambda(lambda x: {"text": x})
#     | template2
#     | model
#     | parser
# )

chain = template1 | model | parser | template2 | model | parser

response = chain.invoke({"topic": "black hole"})
print(response)
