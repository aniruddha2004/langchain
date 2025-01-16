from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")

def callModel(messages) :
    result = model.invoke(messages)
    return result.content
