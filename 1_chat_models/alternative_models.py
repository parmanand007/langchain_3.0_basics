from langchain_core.messages import SystemMessage, HumanMessage,AIMessage

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")
llm = ChatAnthropic(model="claude-3-ops")
llm = Gemini(model="gpt-4o")


messages = [
    SystemMessage(content = "Solve the following math problem below"),
    HumanMessage(content ="What will be square root of 562"),
    AIMessage()
]
