from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd


model = OllamaLLM(model="llama3.2")

df = pd.read_csv("realistic_restaurant_reviews.csv")

review = df["Review"].tolist()
date = df["Date"].tolist()
rating = df["Rating"].tolist()
title = df["Title"].tolist()


template = """
You are an AI built to review pizza restaurants poorly 

Here are some reviews
{reviews}
Here are some date
{dates}
Here are some rating
{ratings}
Here are some restaurant
{titles}

Here are some questions to answer 
{questions}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    question_value = input("press q to quit \nPrompt: ")
    
    if question_value == "q" :
        break
    else:
        result = chain.invoke({
            "ratings": rating, 
            "dates": date, 
            "reviews": review, 
            "titles": title,
            "questions": question_value
            })
        
        print(result)