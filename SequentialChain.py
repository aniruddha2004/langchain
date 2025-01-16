from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda, RunnableSequence

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")


def userInput(x) :
    language = input("Enter the language : ")
    food = input("Enter the name of the food : ")
    return { "language" : language, "food" : food}

sysTemplate = "Translate the following text into {language}"

promptTemplate = ChatPromptTemplate.from_messages([
    ("system", sysTemplate),
    ("human", "I love eating {food}")
])

runnable1 = RunnableLambda(userInput)
runnable2 = RunnableLambda(lambda x : promptTemplate.invoke(x))
runnable3 = RunnableLambda(lambda x : model.invoke(x))
runnable4 = RunnableLambda(lambda x : x.content)

# chain = RunnableSequence(first = runnable1, middle = [runnable2, runnable3], last = runnable4)
chain = runnable1 | runnable2 | runnable3 | runnable4

result = chain.invoke(1)

print(result)