from langchain_community.document_loaders import PyPDFLoader

# from langchain_core.output_parsers import StrOutputParser

# different pdf loaders :
# https://docs.langchain.com/oss/python/integrations/document_loaders/index#pdfs
# https://github.com/undacmic/langchain-pdf-inspector
loader = PyPDFLoader("./dl-curriculum.pdf")
pdf = loader.load()

print(len(pdf))
print(pdf[0].page_content)
