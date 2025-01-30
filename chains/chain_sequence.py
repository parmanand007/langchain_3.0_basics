from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

# Define the prompt templates
animal_fact_template = ChatPromptTemplate.from_messages([
    ("system", "You are a facts expert who knows facts about {animal}."),
    ("human", "Tell me {fact_count} facts.")
])

translation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a translator and convert the provided text into {language}."),
    ("human", "Translate the following text to {language}: {text}")
])

# Define runnable components
count_words = RunnableLambda(lambda x: f"Word Count: {len(x.split())}\n{x}")

prepare_for_translation = RunnableLambda(
    lambda output: {"text": output, "language": "French"}  # Add translation input
)

# Create the chain using the | operator
chain = (
    animal_fact_template
    | llm
    | StrOutputParser()
    | prepare_for_translation
    | translation_template
    | llm
    | StrOutputParser()
)

# Invoke the chain
result = chain.invoke({"animal": "rose", "fact_count": "2"})

# Print the result
print(result)
