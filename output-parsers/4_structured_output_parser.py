from dotenv import load_dotenv
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# In this we provide a json schema to the LLM, and based on the schema it gives us the response
# but now the StructuredOutputParsers are no longer in langchain it is moved to langchain_classic
# so need to install langchain-classic separately

load_dotenv()

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", task="text-generation")

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact 1", description="fact 1 about the topic"),
    ResponseSchema(name="fact 2", description="fact 2 about the topic"),
    ResponseSchema(name="fact 3", description="fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give me 3 facts about the {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = template | model | parser
response = chain.invoke(
    {
        "topic": "Sherlock Holmes",
    }
)

print(response)

# prompt = template.invoke(
#     {
#         "topic": "Shelock Holmes",
#     }
# )

# response = model.invoke(prompt)
# parseResponse = parser.parse(response.content)
# print(parseResponse)
# print(response.content)
