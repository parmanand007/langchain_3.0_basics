from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# Step 1: Initialize the LLM
llm = ChatOpenAI(model="gpt-4")

# Step 2: Define a sentiment analysis function (mock implementation)
def analyze_sentiment(text):
    # Replace this with a real sentiment analysis model
    if "good" in text.lower() or "great" in text.lower():
        return "positive"
    else:
        return "negative"

# Step 3: Define response templates for positive and negative feedback
positive_template = PromptTemplate(
    input_variables=["input"],
    template="Thank you for your positive feedback! We're thrilled to hear that you're enjoying our service. 😊"
)

negative_template = PromptTemplate(
    input_variables=["input"],
    template="We're sorry to hear about your experience. Please let us know how we can improve. Your feedback is valuable to us. 🙏"
)

# Step 4: Create chains for positive and negative feedback
positive_chain = positive_template | (lambda x: {"input": x}) | llm | StrOutputParser()
negative_chain = negative_template | (lambda x: {"input": x}) | llm | StrOutputParser()

# Step 5: Define the Runnable Branch
branch = RunnableBranch(
    branches={
        lambda x: analyze_sentiment(x["input"]) == "positive": positive_chain,
        lambda x: analyze_sentiment(x["input"]) == "negative": negative_chain
    }
)

# Step 6: Wrap the branch in a RunnableLambda for execution
feedback_agent = RunnableLambda(lambda x: branch.invoke(x))

# Step 7: Test the feedback agent
user_input_1 = {"input": "The service was great!"}
response_1 = feedback_agent.invoke(user_input_1)
print(response_1)  # Output: Thank you for your positive feedback! We're thrilled to hear that you're enjoying our service. 😊

user_input_2 = {"input": "I had a bad experience."}
response_2 = feedback_agent.invoke(user_input_2)
print(response_2)  # Output: We're sorry to hear about your experience. Please let us know how we can improve. Your feedback is valuable to us. 🙏
