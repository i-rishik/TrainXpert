from flask import Flask, render_template, request, jsonify
from src.helper import download_google_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import speech_recognition as sr

app = Flask(__name__)

# Load environment variables
load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set environment variables
os.environ["PINECONE_API_KEY"] = pinecone_api_key
os.environ["GOOGLE_API_KEY"] = google_api_key

# Load embeddings and vector store
embeddings = download_google_embeddings()
index_name = "fitbot-index"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Set up LLM and retrieval chain
llm_pipeline = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
document_chain = create_stuff_documents_chain(llm_pipeline, prompt)
rag_chain = create_retrieval_chain(retriever, document_chain)

# Speech recognition function
def take_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.pause_threshold = 1
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio, language="en-in")
        return query
    except Exception:
        return "None"

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("msg")
    response = rag_chain.invoke({"input": msg})
    return jsonify({"answer": response["answer"]})

@app.route("/voice", methods=["POST"])
def voice_chat():
    query = take_command()
    
    if query == "None":
        return jsonify({"query": "None", "answer": "Could not understand. Try again."})
    
    response = rag_chain.invoke({"input": query})
    return jsonify({"query": query, "answer": response["answer"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
