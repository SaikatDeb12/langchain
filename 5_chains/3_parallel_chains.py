from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# RunnableParallel to execute chains parallelly

load_dotenv()

# generating notes on a text, and then generating some quiz based on the notes

llm = HuggingFaceEndpoint(model="google/gemma-4-31B-it")
llm2 = HuggingFaceEndpoint(model="deepseek-ai/DeepSeek-R1")

model = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm2)

prompt = PromptTemplate(
    template="Generate a simple and short notes on the text : \n {text}",
    input_variables=["text"],
)
prompt2 = PromptTemplate(
    template="Generate 3 quiz question on these notes : \n {text}",
    input_variables=["text"],
)
prompt3 = PromptTemplate(
    template="Merge the notes and the quiz into one document. \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

# here we can add any number of chains(with "name") that has to be executed parallelly
parallelChain = RunnableParallel(
    {
        "notes": prompt | model | parser,
        "quiz": prompt2 | model2 | parser,
    }
)

mergeChain = prompt3 | model | parser

chain = parallelChain | mergeChain

notesText = """
                Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.
                The advantages of support vector machines are:
                Effective in high dimensional spaces.
                Still effective in cases where number of dimensions is greater than the number of samples.
                Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
                Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
                The disadvantages of support vector machines include:
                If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
                SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
                The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
            """
response = chain.invoke({"text": notesText})

print(response)

# Response :
# # Support Vector Machines (SVMs)
#
# ## Study Notes
#
# **Overview**
# *   Supervised learning methods used for: **Classification, Regression, and Outlier Detection**.
#
# **Advantages**
# *   **High Dimensionality:** Effective even when features exceed the number of samples.
# *   **Memory Efficient:** Uses only a subset of training points (**Support Vectors**) for decision-making.
# *   **Versatile:** Supports various kernel functions (built-in or custom).
#
# **Disadvantages**
# *   **Overfitting Risk:** Requires careful selection of kernels and regularization when features heavily outweigh samples.
# *   **No Direct Probabilities:** Probability estimates are computationally expensive (require cross-validation).
#
# **Technical Implementation (scikit-learn)**
# *   **Input Types:** Supports both **dense** (numpy) and **sparse** (scipy) vectors.
# *   **Requirement:** To predict sparse data, the model must be trained on sparse data.
# *   **Best Performance:** Use `float64` with C-ordered `numpy.ndarray` or `scipy.sparse.csr_matrix`.
#
# ***
#
# ## Knowledge Check
#
# **Question 1:**
# Which of the following is an advantage of Support Vector Machines (SVMs)?
# A. They naturally output probability estimates without additional computation.
# B. They are effective in high dimensional spaces and when the number of dimensions exceeds the number of samples.
# C. They use all training points in the decision function, leading to higher accuracy.
# D. They are not versatile because only a fixed set of kernel functions can be used.
#
# **Question 2:**
# What is a disadvantage of Support Vector Machines (SVMs)?
# A. They are ineffective in high dimensional spaces.
# B. They require the number of samples to be much greater than the number of features to avoid over-fitting.
# C. They do not directly provide probability estimates; these require an expensive cross-validation step.
# D. They are memory inefficient because they use all training points in the decision function.
#
# **Question 3:**
# According to the notes, which of the following is true about using SVMs in scikit-learn?
# A. SVMs can only use dense data (numpy arrays) as input.
# B. For sparse data, the SVM must be fit on sparse data to make predictions on sparse data.
# C. For optimal performance with dense data, use lists of lists with dtype=float32.
