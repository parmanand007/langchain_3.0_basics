import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from dotenv import load_dotenv

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Initialize the LLM (ChatGPT-4)
llm = ChatOpenAI(model="gpt-4")

# Step 1: Define a template for summarizing the movie
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}")
    ]
)

# Step 2: Define a function to analyze the plot
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?")
        ]
    )
    return plot_template | llm | StrOutputParser()

# Step 3: Define a function to analyze the characters
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?")
        ]
    )
    return character_template | llm | StrOutputParser()

# Step 4: Combine the results of plot and character analysis
def combine_verdicts(results):
    return f"Plot Analysis:\n{results['plot']}\n\nCharacter Analysis:\n{results['characters']}"

# Step 5: Define the chain
chain = (
    summary_template  # Start with the summary template
    | llm  # Generate the movie summary
    | StrOutputParser()  # Parse the output
    | RunnableParallel(  # Run the next two tasks in parallel
        plot=lambda x: analyze_plot(x).invoke({"plot": x}),  # Analyze the plot
        characters=lambda x: analyze_characters(x).invoke({"characters": x})  # Analyze the characters
    )
    | combine_verdicts  # Combine the results
)

# Step 6: Invoke the chain with a movie name
result = chain.invoke({"movie_name": "Inception"})

# Step 7: Print the result
print(result)