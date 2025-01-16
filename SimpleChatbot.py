from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import chat
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore
# cred = credentials.Certificate("path/to/serviceAccountKey.json")  # Replace with your key path
cred = credentials.Certificate("assistantnode-79573-firebase-adminsdk-oz9yv-d8444ab22a.json")  # Replace with your key path
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firestore collection name
COLLECTION_NAME = "chat_history"

# Helper function to serialize a message
def serialize_message(message):
    if isinstance(message, SystemMessage):
        return {"type": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        return {"type": "human", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"type": "ai", "content": message.content}
    else:
        raise ValueError(f"Unknown message type: {type(message)}")

# Helper function to deserialize a message
def deserialize_message(message_dict):
    msg_type = message_dict["type"]
    content = message_dict["content"]

    if msg_type == "system":
        return SystemMessage(content=content)
    elif msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")

# Store a serialized message in Firestore
def store_message(user_id, message):
    doc_ref = db.collection(COLLECTION_NAME).document(user_id)
    chat_doc = doc_ref.get()

    serialized_message = serialize_message(message)
    
    if chat_doc.exists:
        chat_data = chat_doc.to_dict()
        chat_data["messages"].append(serialized_message)
        doc_ref.set(chat_data)
    else:
        doc_ref.set({"messages": [serialized_message]})

# Retrieve and deserialize messages from Firestore
def get_chat_history(user_id):
    doc_ref = db.collection(COLLECTION_NAME).document(user_id)
    chat_doc = doc_ref.get()
    if chat_doc.exists:
        serialized_messages = chat_doc.to_dict().get("messages", [])
        return [deserialize_message(msg) for msg in serialized_messages]
    return []

# Validate the sequence of chat history
def validate_chat_history(chat_history):
    # Ensure the first message is a SystemMessage
    if not isinstance(chat_history[0], SystemMessage):
        raise ValueError("The first message must be a SystemMessage.")
    
    # Check for alternating HumanMessage and AIMessage
    for i in range(1, len(chat_history)):
        if i % 2 == 1 and not isinstance(chat_history[i], HumanMessage):
            raise ValueError(f"Expected HumanMessage at position {i}, found {type(chat_history[i])}.")
        elif i % 2 == 0 and not isinstance(chat_history[i], AIMessage):
            raise ValueError(f"Expected AIMessage at position {i}, found {type(chat_history[i])}.")

# Start the chatbot
user_id = "unique_user_id"  # Replace with a dynamic user identifier if needed
chatHistory = get_chat_history(user_id) or []

# Ensure only one SystemMessage
if not any(isinstance(msg, SystemMessage) for msg in chatHistory):
    chatHistory.insert(0, SystemMessage("You are a chat model who gives answers in Shakespearian English"))

while True:
    query = input("INPUT : ")
    if query.lower() == "exit":
        print("ARCELY : Thank you so much for chatting with me! Have a nice day.")
        break

    # Add human message to chat history
    human_message = HumanMessage(content=query)
    chatHistory.append(human_message)
    store_message(user_id, human_message)

    # Validate chat history before calling the model
    try:
        validate_chat_history(chatHistory)
        result = chat.callModel(chatHistory)
    except ValueError as e:
        print(f"Error in chat history: {e}")
        break

    # Add AI message to chat history
    ai_message = AIMessage(content=result)
    chatHistory.append(ai_message)
    store_message(user_id, ai_message)

    # Print AI response
    print("ARCELY : " + result)
