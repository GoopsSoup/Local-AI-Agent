from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
from vector import retriever


model = OllamaLLM(model="llama3.2")

template = """
You are an AI built to review pizza restaurants with poor performance and actually look likes someone that never reviewed any pizza restaurant 

Here are some reviews
{reviews}

Here are some questions to answer 
{questions}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question_value = input("press q to quit \nPrompt: ")
    
    if question_value == "q" :
        break
    
    reviews = retriever.invoke(question_value)
    reviews_text = "\n\n".join([doc.page_content for doc in reviews])
    
    result = chain.invoke({
        "reviews": reviews_text, 
        "questions": question_value
        })
        
    print(result)