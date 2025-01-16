from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
# from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

embed = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY"),
)

# loader = TextLoader("story.txt")
# documents = loader.load()

# textSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
# splittedDocuments = textSplitter.split_documents(documents)

# print("No. of chunks = ", len(splittedDocuments))
# print("\n\n\n\nThe first chunk = " + splittedDocuments[0].page_content)

# vector = embed.embed_query(text)
# print(vector)

storyStore2 = Chroma(
    collection_name="storyCollection",
    embedding_function=embed,
    persist_directory="./trial_db",  # Where to save data locally, remove if not necessary
)

# storyStore2.add_documents(documents = splittedDocuments)


# documents = [Document(page_content = "Elon musk is a billionaire", metadata = {"source" : "tweet"}), Document(page_content = "Bill Gates is a billionaire", metadata = {"source" : "news"}), Document(page_content = "Mukesh Ambani is a billionaire", metadata = {"source" : "news"}), Document(page_content = "Jeff Bezos is a billionaire", metadata = {"source" : "tweet"})]

# vector_store.add_documents(documents = documents, ids = ["B01", "B02", "B03", "B04"])

# vector_store.update_documents(ids = ["B01", "B02", "B03", "B04"], documents = documents)

# Retrieve all documents from the vector store
# retrieved_data = storyStore2._collection.get()
# print("Inserted Data:", retrieved_data)

results = storyStore2.similarity_search("What is an example of a large solar farm?", k=1)
for result in results :
    print(result.page_content)
