import random
from abc import ABC, abstractmethod

# using Runnables to standarize the components
# by making an abstractmethod named 'invoke' and then implementing the method for all the different components


class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass


class NakliLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence",
        ]

        return {"response": random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence",
        ]

        return {"response": random.choice(response_list)}


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, input_data):
        return input_data["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data


template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}", input_variables=["length", "topic"]
)

prompt = template.format({"length": "short", "topic": "india"})
llm = NakliLLM()

llm.predict(prompt)


template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}", input_variables=["length", "topic"]
)

llm = NakliLLM()
chain = NakliLLMChain(llm, template)

chain.run({"length": "short", "topic": "india"})
