from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# temperature controls the randomness of the model
# 0 - 0.3 -> math, code, factual answer
# 0.5 - 0.7 -> general QA
# 0.9 - 1.2 -> creative writing, storytelling
# 1.5+ -> max randomness
model = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=1.0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

response = model.invoke("tell me a short and funny story")  # temp : 1.6
print(response.text)
