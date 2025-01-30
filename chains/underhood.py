from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence
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

# without calling invoke
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

invoke_model = RunnableLambda(lambda x: llm.invoke(x.to_messages()))

parse_output = RunnableLambda(lambda x: x.content)




chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)
# or 
# chain = format_prompt | invoke_model | parse_output

result = chain.invoke({"animal":"cow","fact_count":2})

print(result)