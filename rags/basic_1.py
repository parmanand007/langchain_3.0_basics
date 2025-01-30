from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# define dir

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chorma_db")
# check chroma vector store

if not os.path.exists(persistent_directory):
    print("Initializing vector store..")

    # Ensure text file exist

    if not os.path.exists(file_path):
        raise FileNotFoundError(" The file {file_path} doesn't exist.")

    # Read the text content ffrom the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # display info of split documents
    print("\n--- Documents Chunk Information ----")
    print(f"Number of document chunks : {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # create embeddings

    print("\n--- Creating embeddings ---\n")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n--- Finished creating embeddings---")

    # create the vector store and persist it automatically

    print("\n--- Creating vectory store---\n")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)

else:
    print("Vector store already exists. No need to initalize")
