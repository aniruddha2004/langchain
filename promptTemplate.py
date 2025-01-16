from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")

# temp = "Translate the following from English into {language} : {text}"
sysTemplate = "Translate the following from English into {language}"

message = [
    ("system", sysTemplate),
    ("human", "I love my country")
]

# promptTemplate = ChatPromptTemplate.from_template(temp)
promptMessage = ChatPromptTemplate.from_messages(message)

# prompt = promptMessage.invoke({
#     "language" : "Bengali",
#     "text" : "My country is India."
# })

chain = promptMessage | model 

result = chain.invoke({
    "language" : "Bengali",
    "text" : "My country is India."
})


# result = model.invoke(prompt)
print(result.content)