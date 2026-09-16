from langchain_core.prompts import PromptTemplate

# PromptTemplate an object
# instead of using PromptTemplate we can also use fstring as well
# but, PromptTemplate allows us to define placeholders
# the number of input parameters that were used
# and it throws error when the placeholders are not given as input

template = PromptTemplate(
    template="""
    Summarize the reasearch paper titled {paper_input} with the following specifications:
    Explaination style : {paper_style}
    Explaination length : {paper_length}
    1. Mahtematical details :
        - Include relevant mathematical equations if present in the paper.
        - Explain the mathematical concepts using simple, intuitive code snippets where applicable.
    2. Analogies:
        - Use relatable analogies to simplify complex ideas.
    If certian infomation is not available in the paper, respond with "Insufficient information vavailable" instead of guessing.
    Ensure the summary is clar, accurate, and aligned with the provied style and length.
    """,
    input_variables=["paper_input", "paper_style", "paper_length"],
    validate_template=True,
)

template.save("template.json")
