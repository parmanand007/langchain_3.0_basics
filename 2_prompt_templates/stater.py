from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
import os


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

template = "Write a {tone} email to {company} expressing intrest in the {position}, mentioning {skill} as key strength , keep it to 4 line max"


prompt_template = ChatPromptTemplate.from_template(template)

prompt =prompt_template.invoke({
    "tone":"energetic",
    "company":"dataplant",
    "position":"backend developer",
    "skill":"AI"
})


# print(prompt_template)
# print(prompt)

# response=llm.invoke(prompt)
# print(response.content)


# Example 2


messages = [
    ("system", "you are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})

print(prompt)


response = llm.invoke(prompt)
print(response.content)