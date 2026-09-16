import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

st.header("Research Summarizer")

model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

# input = st.text_input("enter your prompt")
paper_input = st.selectbox(
    "Select research paper name",
    [
        "Attention is all you need",
        "BERT: Pre-training of deep bidirectional transformers",
        "GPT-3: Language models are few-shot learners",
        "Diffusion models beat GANs on image synthesis",
    ],
)

paper_style = st.selectbox(
    "Select explaination style",
    ["Beginner-friendly", "Techinical", "Code-Oriented", "Mathematics"],
)

paper_length = st.selectbox(
    "Select explaination length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explaination)",
    ],
)

template = load_prompt("./template.json")

if st.button("Summarize"):
    chain = template | model
    response = chain.invoke(
        {
            "paper_input": paper_input,
            "paper_style": paper_style,
            "paper_length": paper_length,
        }
    )
    st.write(response.content[0]["text"])

# template = load_prompt("./template.json")
#
# # fill in the placeholders
# prompt = template.invoke(
#     {
#         "paper_input": paper_input,
#         "paper_style": paper_style,
#         "paper_length": paper_length,
#     }
# )
#
# if st.button("Summarize"):
#     response = model.invoke(prompt)
#     st.write(response.content[0]["text"])
#
