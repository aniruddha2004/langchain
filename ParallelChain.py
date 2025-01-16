from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")

sysPrompt = "You are a movie critic"

promptMessage = ChatPromptTemplate.from_messages([
    ("system", sysPrompt),
    ("human", "Generate a very brief summary for the movie : {movie}")
])

def userInput(x) : 
    movie = input("Enter the movie name : ")
    return {"movie" : movie}

storyTemp = ChatPromptTemplate.from_template("Give the story analysis for the movie summary : {summary}")
characterTemp = ChatPromptTemplate.from_template("Give character analysis for the movie summary : {summary}")

inputMovie = RunnableLambda(userInput)
createPrompt = RunnableLambda(lambda x : promptMessage.invoke(x))
generateSummary = RunnableLambda(lambda x : model.invoke(x))
getContent = RunnableLambda(lambda x : {"summary" : x.content})
storyAnalysisChain = RunnableLambda(lambda x : storyTemp.invoke(x)) | RunnableLambda(lambda x : model.invoke(x)) | RunnableLambda(lambda x : x.content)
characterAnalysisChain = RunnableLambda(lambda x : characterTemp.invoke(x)) | RunnableLambda(lambda x : model.invoke(x)) | RunnableLambda(lambda x : x.content)
getResult = RunnableLambda(lambda x : {"sa" : x['branches']['story'], "ca" : x['branches']['character']})
chain = inputMovie | createPrompt | generateSummary | getContent | RunnableParallel(branches = {"story" : storyAnalysisChain, "character" : characterAnalysisChain}) | getResult

result = chain.invoke(1)

print(result['sa'])
print(result['ca'])




