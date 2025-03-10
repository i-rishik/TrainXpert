# TrainXpert - AI-Powered Fitness Chatbot

TrainXpert is an advanced AI-powered chatbot designed to provide fitness-related insights using Retrieval-Augmented Generation (RAG) techniques. By combining Google Gemini’s powerful text-generation capabilities with Pinecone’s vector-based retrieval system, TrainXpert delivers precise, context-aware answers to fitness queries.



https://github.com/user-attachments/assets/c8bd385c-a7e7-4fc5-9ed0-f2a5d038926e



## 📂 Project Structure  
📂 TrainXpert  
 ┣ 📂 src  
 ┃ ┣ 📜 helper.py  
 ┃ ┣ 📜 prompt.py  
 ┣ 📂 static  
 ┃ ┣ 📜 styles.css  
 ┣ 📂 templates  
 ┃ ┣ 📜 index.html  
 ┣ 📜 .env  
 ┣ 📜 app.py  
 ┣ 📜 requirements.txt  
 ┣ 📜 setup.py  
 ┣ 📜 vector_store.py  
 ┣ 📜 README.md  
 ┣ 📜 .gitignore  

 ## ✨ Features  
✅ **Natural Language Understanding:** Uses `Google Generative AI` for intelligent responses.  
✅ **Voice Query Support:** Users can ask fitness-related questions using speech input.  
✅ **Contextual Answers:** Retrieves information from vector storage using `Pinecone`.  
✅ **Interactive Web Interface:** Built with `Flask` and styled using Bootstrap.  
✅ **PDF Integration:** Loads fitness-related PDFs for contextual retrieval.  

## 📄 Project documentation  
### Installation
#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/TrainXpert.git
cd TrainXpert  
```

#### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Set up API Keys
Create a .env file in the root directory and add:
```bash
PINECONE_API_KEY=your_pinecone_key
GOOGLE_API_KEY=your_google_key
```

#### 5. Create a new pinecone index (Only when new index had to be created or new pdf is used)
```bash
python vector_store.py
```

#### 6. Running the chatbot
```bash
python app.py
```
Open http://localhost:5000/ in your browser

## 📌 Working
1️⃣ User Input: The chatbot now supports both text and voice input via the web UI.  
2️⃣ Retrieval: It searches for relevant context in Pinecone.  
3️⃣ Response Generation: It feeds the retrieved documents to Google Gemini for answer generation.  
4️⃣ Display Response: The chatbot returns a concise, relevant answer.

## 🛠️ Tech Stack  
**Flask** – Backend API  
**SpeechRecognition** - Converts voice input to text  
**LangChain** – Chaining retrieval and response generation  
**Pinecone** – Vector-based retrieval  
**Google Generative AI Embeddings** – Embeddings for document storage  
**HTML & CSS** – Basic UI with CSS styling  
