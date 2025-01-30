from langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from google.cloud import firestore
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreChatMessageHistory
from dotenv import load_dotenv
load_dotenv()
import os


system_message = SystemMessage(content="You are a helpful AI assistant")


PROJECT_ID = os.getenv("PROJECT_ID")
SESSION_ID = os.getenv("SESSION_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

print("Initalizing firestore client....")
client = firestore.Client(project=PROJECT_ID)


# Initialize firestore chat message history

print("Initializing firestore chat message History.....")

chat_history = FirestoreChatMessageHistory(session_id=SESSION_ID,collection=COLLECTION_NAME,
                                           client=client)


print("chat history is initalized...")
print("Current chat history:",chat_history.messages)


print("Start chatting with open ai")
llm = ChatOpenAI(model="gpt-4o")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))
    
    result = llm.invoke(chat_history)

    response = result.content
    print(response)