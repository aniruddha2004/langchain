from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()

embed = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=os.getenv("MISTRAL_API_KEY"),
)

story = Chroma(
    collection_name="storyCollection",
    embedding_function=embed,
    persist_directory="./trial_db",  # Where to save data locally, remove if not necessary
)

retreiver = story.as_retriever(search_type = "similarity_score_threshold", search_kwargs = {"k" : 1, "score_threshold" : 0.5})

model = ChatMistralAI(model="mistral-large-latest")

query = input("Enter your question : ")

systemPrompt = "You are a helpful AI assitant"
humanPrompt = "Here is a piece of information '{content}'. Answer the question '{query}' from within the provided contents only."

promptTemplate = ChatPromptTemplate.from_messages([
    ("system", systemPrompt),
    ("human", humanPrompt)
])

def extractor(x) : 
    contents = [doc.page_content for doc in x]
    return {"query" : query, "content" : contents[0]}

chain = retreiver | RunnableLambda(extractor) | promptTemplate | model | RunnableLambda(lambda x : x.content)

result = chain.invoke(query)

print(result)