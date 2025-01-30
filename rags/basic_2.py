import os
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings

# define the persistent directory

current_dir = os.path.dirname(__file__)
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# define the embedding model

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load the existing vector store with embeddings functions

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# define user's query

query = "Where does Gandalf meet Frodo ?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1}


)


relevant_doc = retriever.invoke(query)
# print(relevant_doc)
# display

print("\n====== Relevant documents============\n")

for i, doc in enumerate(relevant_doc, 1):
    print(f"Document {i}: \n {doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source','Unknown')}\n")
