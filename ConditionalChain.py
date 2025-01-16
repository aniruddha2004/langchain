from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda, RunnableBranch

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")

systemPrompt = "You are a customer feedback analyzer, classify a feedback as positive or negative"

promptMessages = ChatPromptTemplate.from_messages([
    ("system", systemPrompt),
    ("human", "Here is the feedback from the customer: {feedback}")
])

def userInput(x) : 
    review = input("Give your feedback : ")
    return {"feedback" : review}

# takeInput = RunnableLambda(userInput)
# createPrompt = RunnableLambda(lambda x : promptMessages.invoke(x))
# analyzeFeedback = RunnableLambda(lambda x : model.invoke(x))
# getResult = RunnableLambda(lambda x : x.content)

# chain = takeInput | createPrompt | analyzeFeedback | getResult

analyzeFeedbackChain = RunnableLambda(userInput) | promptMessages | model | RunnableLambda(lambda x : {"feedback" : x.content.lower()})

# analyzeFeedbackResult = analyzeFeedbackChain.invoke(1)
# print(analyzeFeedbackResult)

positiveResponse = ChatPromptTemplate.from_template("Generate a positive response based on the feedback : {feedback}")
negativeResponse = ChatPromptTemplate.from_template("Generate a apologetic response based on the negative feedback : {feedback}")

positiveResponseChain = positiveResponse | model | RunnableLambda(lambda x : x.content)
negativeResponseChain = negativeResponse | model | RunnableLambda(lambda x : x.content)

branches = RunnableBranch(
    (
        lambda x : "positive" in x["feedback"],
        positiveResponseChain
    ),
    (
        lambda x : "negative" in x["feedback"],
        negativeResponseChain
    ),
    lambda x : "goodbye"
)

resultChain = analyzeFeedbackChain | branches

result = resultChain.invoke(1)
print(result)