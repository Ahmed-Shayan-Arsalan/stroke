from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langdetect import detect
import os

# Initialize Flask app
app = Flask(__name__)

# Set up your Groq API key
groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# User data dictionary to store names
user_data = {}

translations = {
    "en": {
        "info": "Here's the information you requested:\n\n",
        "support": "To support you, here's what I found:\n\n",
        "taskhelp": "Here's a guide to help you with your task:\n\n",
        "video": "Here's a relevant video or related information:\n\n",
        "context": "in Malaysia"
    },
    "zh": {
        "info": "这是您请求的信息：\n\n",
        "support": "为了支持您，这是我发现的内容：\n\n",
        "taskhelp": "这是帮助您完成任务的指南：\n\n",
        "video": "这是相关视频或相关信息：\n\n",
        "context": "在马来西亚"
    },
    "ms": {
        "info": "Inilah maklumat yang anda minta:\n\n",
        "support": "Untuk menyokong anda, berikut adalah apa yang saya dapati:\n\n",
        "taskhelp": "Berikut adalah panduan untuk membantu anda dengan tugas anda:\n\n",
        "video": "Berikut adalah video atau maklumat yang berkaitan:\n\n",
        "context": "di Malaysia"
    },
    "ta": {
        "info": "நீங்கள் கேட்ட தகவல் இங்கே:\n\n",
        "support": "உங்களுக்கு ஆதரவு அளிக்க, நான் கண்டறிந்தது இங்கே:\n\n",
        "taskhelp": "உங்கள் பணியில் உதவ ஒரு வழிகாட்டி இங்கே:\n\n",
        "video": "இங்கே ஒரு தொடர்புடைய வீடியோ அல்லது தொடர்புடைய தகவல்:\n\n",
        "context": "மலேசியாவில்"
    }
}

class ChatbotAgent:
    def __init__(self):
        self.qa_interface = self.setup_chatbot()

    def setup_chatbot(self):
        pdf_paths = [
            "./data/caknaStroke2.pdf",
            "./data/FinancialAids.pdf",
            "./data/strokes.pdf",
            "./data/Penjagaan_Pesakit_Strok_Translated.pdf"
        ]

        combined_text = ""
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            pdf_text = loader.load()
            combined_text += "\n\n".join([doc.page_content for doc in pdf_text]) + "\n\n"

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.create_documents([combined_text])

        directory = "./data/index_store"
        vector_index = FAISS.from_documents(
            texts,
            HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        )
        vector_index.save_local(directory)

        vector_index = FAISS.load_local(
            directory,
            HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
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

    def process_query(self, query):
        result = self.qa_interface.invoke({"query": query})
        return result["result"]

    def extract_name_with_llm(self, user_input):
        """
        Uses LLM to extract the name from user input.
        """
        # Craft a clear and concise instruction for the LLM
        extraction_prompt = f"Extract the name from this sentence: '{user_input}'. If there is no name, respond with 'None'. Keep in mind, just provide the name and nothing else."

        # Pass the prompt directly to the LLM
        response = llm.invoke(extraction_prompt)

        # Access the 'content' attribute of the response
        extracted_name = response.content.strip()  # Get the text and remove extra whitespace

        # Return None if the response explicitly says 'None'
        return None if extracted_name.lower() == "none" else extracted_name

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"  # Default to English if detection fails

# Initialize chatbot agent
chatbot_agent = ChatbotAgent()

@app.route("/webhook", methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').strip()
    phone_number = request.values.get('From', '')

    # Detect the language of the incoming message
    detected_language = chatbot_agent.detect_language(incoming_msg)
    language_translations = translations.get(detected_language, translations["en"])

    # Check if the user's name is stored
    if phone_number not in user_data:
        user_data[phone_number] = {"name": None}

    if user_data[phone_number]["name"] is None:
        if incoming_msg.lower() in ["hi", "hey", "hello"]:  # Initial greeting
            # Ask for the user's name in the detected language
            name_request = {
                "en": "Hi there! Before we continue, could you please tell me your name?",
                "zh": "您好！在继续之前，您能告诉我您的名字吗？",
                "ms": "Hai! Sebelum kita teruskan, bolehkah anda beritahu saya nama anda?",
                "ta": "வணக்கம்! தொடருவதற்கு முன் உங்கள் பெயரை சொல்ல முடியுமா?"
            }
            twilio_response = MessagingResponse()
            twilio_response.message(name_request.get(detected_language, name_request["en"]))
            return str(twilio_response)
        else:
            # Use LLM to extract the user's name
            extracted_name = chatbot_agent.extract_name_with_llm(incoming_msg)
            if extracted_name:
                user_data[phone_number]["name"] = extracted_name
                thank_you_message = {
                    "en": f"Thank you, {extracted_name}! How can I assist you today?",
                    "zh": f"谢谢您, {extracted_name}！今天我能帮您什么？",
                    "ms": f"Terima kasih, {extracted_name}! Bagaimana saya boleh membantu anda hari ini?",
                    "ta": f"நன்றி, {extracted_name}! இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?"
                }
                twilio_response = MessagingResponse()
                twilio_response.message(thank_you_message.get(detected_language, thank_you_message["en"]))
                return str(twilio_response)
            else:
                # If no name could be extracted, ask again
                twilio_response = MessagingResponse()
                name_request = {
                    "en": "I couldn't quite catch your name. Could you please provide it again?",
                    "zh": "我没能听清您的名字。请您再提供一次好吗？",
                    "ms": "Saya tidak dapat menangkap nama anda. Bolehkah anda berikan sekali lagi?",
                    "ta": "நான் உங்கள் பெயரைப் பிடிக்க முடியவில்லை. தயவுசெய்து மீண்டும் சொல்லுங்கள்."
                }
                twilio_response.message(name_request.get(detected_language, name_request["en"]))
                return str(twilio_response)

    # Use the stored name in responses
    user_name = user_data[phone_number]["name"]

    # Process the query through the chatbot agent
    response_text = chatbot_agent.process_query(incoming_msg)

    # Format the response to include the user's name
    formatted_response = f"{user_name}, {response_text}"

    # Format the response for Twilio
    twilio_response = MessagingResponse()

    # Split long responses into chunks
    if len(formatted_response) > 1500:
        chunks = [formatted_response[i:i + 1500] for i in range(0, len(formatted_response), 1500)]
        for chunk in chunks:
            twilio_response.message(chunk)
    else:
        twilio_response.message(formatted_response)

    return str(twilio_response)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
