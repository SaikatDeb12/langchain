from typing import Literal

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel, Field

# RunnableBranch is used to run chians on condition
load_dotenv()

# based on the feedback we want to find out the sentiment of the feedback, either (+)ve or (-)ve

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


# prompt tells the LLM what task to perform; Field(description=...) tells the structured-output schema what each output field means.
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="classify the sentiment of the following feedback text into positive or negative sentiment: \n {feedback} \n {output_rules}",
    input_variables=["feedback"],
    partial_variables={
        "output_rules": parser2.get_format_instructions(),
    },
)

classifierChain = prompt | model | parser2
# response = classifierChain.invoke({"feedback": "this is an overall good smartphone"})

# but here, we have a well structured output of the sentiment
# print(response.sentiment)

prompt2 = PromptTemplate(
    template="write a short 1 line reply thanking the user : \n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="write a short 1 line reply to apologize the user : \n {feedback}",
    input_variables=["feedback"],
)

# RunnableBranch takes multiple tuples with (condition, which chain to execute)
branchChain = RunnableBranch(
    (lambda x: x["sentiment"].sentiment == "positive", prompt2 | model | parser),
    (lambda x: x["sentiment"].sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "cannot find appropriate sentiment"),
)

chain = RunnablePassthrough.assign(sentiment=classifierChain) | branchChain
response = chain.invoke(
    {"feedback": "this is one of the best smartphone present in the market"}
)
print(response)


# So the output is Positive or Negative, but we want to have the response in a structured way with consistency
# print(response)
