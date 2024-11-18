import os
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import pdfplumber
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load Groq API key from environment variable
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Translation dictionaries
translations = {
    "en": {"info": "Here's the information you requested:\n\n", "support": "To support you, here's what I found:\n\n", "taskhelp": "Here's a guide to help you with your task:\n\n", "video": "Here's a relevant video or related information:\n\n", "context": "in Malaysia"},
    "zh": {"info": "这是您请求的信息：\n\n", "support": "为了支持您，这是我发现的内容：\n\n", "taskhelp": "这是帮助您完成任务的指南：\n\n", "video": "这是相关视频或相关信息：\n\n", "context": "在马来西亚"},
    "ms": {"info": "Inilah maklumat yang anda minta:\n\n", "support": "Untuk menyokong anda, berikut adalah apa yang saya dapati:\n\n", "taskhelp": "Berikut adalah panduan untuk membantu anda dengan tugas anda:\n\n", "video": "Berikut adalah video atau maklumat yang berkaitan:\n\n", "context": "di Malaysia"},
    "ta": {"info": "நீங்கள் கேட்ட தகவல் இங்கே:\n\n", "support": "உங்களுக்கு ஆதரவு அளிக்க, நான் கண்டறிந்தது இங்கே:\n\n", "taskhelp": "உங்கள் பணியில் உதவ ஒரு வழிகாட்டி இங்கே:\n\n", "video": "இங்கே ஒரு தொடர்புடைய வீடியோ அல்லது தொடர்புடைய தகவல்:\n\n", "context": "மலேசியாவில்"}
}

# QueryProcessor Class
class QueryProcessor:
    def __init__(self):
        self.context_keywords = ['hospital', 'doctor', 'clinic', 'healthcare', 'cost', 'treatment', 'medicine', 'support', 'financial', 'aid', 'contact', 'specialist', 'rehabilitation', 'therapy', 'emergency', 'ambulance', 'insurance', 'payment']
        self.theoretical_keywords = ['what is', 'symptoms', 'signs', 'causes', 'risk factors', 'prevention', 'how to identify', 'what happens', 'recovery process', 'brain', 'effects', 'definition', 'types of stroke']
    
    def should_add_context(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.context_keywords)
    
    def is_theoretical_query(self, query):
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.theoretical_keywords)
    
    def process_query(self, query, language):
        if self.should_add_context(query) and not self.is_theoretical_query(query):
            return f"{query} {translations[language]['context']}"
        return query

# ChatbotAgent Class
class ChatbotAgent:
    def __init__(self, pdf_paths):
        self.query_processor = QueryProcessor()
        self.qa_interface = self.setup_chatbot(pdf_paths)
        self.conversation_memory = deque(maxlen=15)

    def setup_chatbot(self, pdf_paths):
        combined_text = ""
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):  # Ensure file exists
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        combined_text += page.extract_text() + "\n\n"
            else:
                logging.error(f"PDF file not found: {pdf_path}")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents([combined_text])

        directory = "data/index_store"
        vector_index = FAISS.from_documents(
            texts,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
        vector_index.save_local(directory)

        vector_index = FAISS.load_local(
            directory,
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True
        )

        retriever = vector_index.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        qa_interface = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        return qa_interface

    def detect_language(self, text):
        if any(word in text for word in ["的", "是", "我", "你", "他", "吗"]):
            return "zh"  
        elif any(word in text for word in ["dan", "untuk", "anda", "saya", "dengan"]):
            return "ms"  
        elif any(word in text for word in ["உங்கள்", "என்", "எப்போதும்", "என்று", "பயன்பாடு"]):
            return "ta"  
        else:
            return "en"  

    def query_qa_interface(self, query, query_type):
        language = self.detect_language(query)
        processed_query = self.query_processor.process_query(query, language)
        memory_context = "\n".join([f"User: {q}\nBot: {r}" for q, r in self.conversation_memory])
        query_with_context = f"{memory_context}\nUser: {processed_query}" if memory_context else processed_query

        try:
            result = self.qa_interface.invoke({"query": query_with_context})
            response = result["result"]
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            response = "I'm sorry, I couldn't process your query."

        header = translations[language].get(query_type, "")
        formatted_response = f"{header}{response}"
        self.conversation_memory.append((query, formatted_response))

        return formatted_response

# Flask App Setup
app = Flask(__name__)
agent = None

@app.before_first_request
def initialize_agent():
    global agent
    pdf_paths = [
        "./data/caknaStroke2.pdf",
        "./data/FinancialAids.pdf",
        "./data/mystrokejourney.pdf",
        "./data/Penjagaan_Pesakit_Strok.pdf"
    ]
    try:
        agent = ChatbotAgent(pdf_paths)
        logging.info("ChatbotAgent initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing ChatbotAgent: {e}")

@app.route("/")
def home():
    return "Welcome to the Chatbot Application!"

@app.route('/webhook', methods=['POST', 'GET'])
def webhook():
    try:
        if request.method == 'GET':
            logging.debug("GET request received at /webhook")
            return "Webhook is active. Use POST requests to interact.", 200

        logging.debug("POST request received at /webhook")
        incoming_msg = request.form.get('Body', '').strip()
        sender = request.form.get('From', '').strip()

        if not incoming_msg:
            return "No message content received.", 400

        logging.debug(f"Incoming message: {incoming_msg} from {sender}")
        query_type = classify_query_type(incoming_msg)
        response = agent.query_qa_interface(incoming_msg, query_type)

        logging.debug(f"Response generated: {response}")
        twilio_resp = MessagingResponse()
        twilio_resp.message(response)
        return str(twilio_resp)

    except Exception as e:
        logging.error(f"Error in webhook: {e}")
        return "Internal server error.", 500

def classify_query_type(query):
    query = query.lower()
    if "video" in query:
        return "video"
    elif "how to" in query or "guide" in query:
        return "taskhelp"
    elif "help" in query or "support" in query:
        return "support"
    else:
        return "info"  

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
