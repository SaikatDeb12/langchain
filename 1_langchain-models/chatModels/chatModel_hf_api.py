from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
)

# chat_model = ChatHuggingFace(llm=llm)
# llm = HuggingFaceEndpoint(
#     repo_id="openai-community/gpt2",
#     task="text-generation",
#     max_new_tokens=120,
# )

model = ChatHuggingFace(llm=llm)
response = model.invoke("what is the capital of India")
print(response.content)
