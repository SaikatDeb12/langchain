from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

# from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="write a very short summary of the following text : \n {text}",
    input_variables=["text"],
)

parser = StrOutputParser()

# another method :
# with open("./cricket.txt", "r", encoding="utf-8") as f:
#     textContent = f.read()
#
# document = Document(page_content=textContent)
# print(document.page_content)

loader = TextLoader("cricket.txt", "utf-8")
# Document object :
# Document(
#     page_content = "",
#     meta_data = ""
# )
fullDoc = loader.load()
content = fullDoc[0].page_content

# the type is always a list
# print(type(doc))

chain = prompt | model | parser
response = chain.invoke(
    {
        "text": content,
    }
)

print(response)
