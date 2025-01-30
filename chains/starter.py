from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
import os


load_dotenv()

llm = ChatOpenAI(model="gpt-4o")


messages = [
    ("system", "you are a facts expert who knows fact about  {animal}"),
    ("human", "Tell me {fact_count} facts.")
]

prompt_template = ChatPromptTemplate.from_messages(messages)


chain = prompt_template | llm | StrOutputParser()


result = chain.invoke({"animal":"cow","fact_count":2})

print(result)