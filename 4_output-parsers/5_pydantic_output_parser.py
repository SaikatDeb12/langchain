from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

# well use pydanticOutputParser for structuredOutput and promptParser with validation
load_dotenv()

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it", task="text-generation")

model = ChatHuggingFace(llm=llm)


class Type(BaseModel):
    name: str = Field(description="name of the person")
    age: int = Field(gt=18, description="age of the person")
    city: str = Field(description="name of the city where the person belongs to")


parser = PydanticOutputParser(pydantic_object=Type)

# partial_variables are used when a value in the prompt is fixed and we dont want to send it everytime we call prompt
template = PromptTemplate(
    template="generate a {type} person along with its age and city \n {output_rules}",
    input_variables=["type"],
    partial_variables={"output_rules": parser.get_format_instructions()},
)

chain = template | model | parser
response = chain.invoke({"type": "US"})
print(response)

# prompt = template.invoke(
#     {
#         "type": "marvel",  # types can be DC, Marvel, cartoon
#     }
# )
#
# print(prompt)
# response = model.invoke(prompt)
# parseResponse = parser.parse(response.content)
# print(parseResponse)
