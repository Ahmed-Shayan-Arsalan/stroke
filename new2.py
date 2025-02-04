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

# User data dictionary to store names and current section
user_data = {}

# Original translations dictionary
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

# Navigation menu with numbers
navigation_menu = {
    "en": """Please select a number to continue:
    1. 🧠 Stroke Education
    2. 💊 Medication Management
    3. 🤝 Emotional Support
    4. 🏋️‍♀️ Rehabilitation Guidance
    5. 🌐 Resources & Contacts
    6. ⚠️ Emergency Response

    Type 'menu' or 'back' to return to this menu
    Type 'help' for section-specific guidance""",
        
        "zh": """请选择一个数字继续：
    1. 🧠 中风教育
    2. 💊 药物管理
    3. 🤝 情感支持
    4. 🏋️‍♀️ 康复指导
    5. 🌐 资源与联系
    6. ⚠️ 紧急响应

    输入 'menu' 或 'back' 返回此菜单
    输入 'help' 获取部分特定指导""",
        
        "ms": """Sila pilih nombor untuk meneruskan:
    1. 🧠 Pendidikan Strok
    2. 💊 Pengurusan Ubat
    3. 🤝 Sokongan Emosi
    4. 🏋️‍♀️ Panduan Pemulihan
    5. 🌐 Sumber & Hubungan
    6. ⚠️ Respons Kecemasan

    Taip 'menu' atau 'back' untuk kembali ke menu ini
    Taip 'help' untuk panduan khusus bahagian""",
        
        "ta": """தொடர ஒரு எண்ணைத் தேர்ந்தெடுக்கவும்:
    1. 🧠 பக்கவாதக் கல்வி
    2. 💊 மருந்து மேலாண்மை
    3. 🤝 உணர்ச்சி ஆதரவு
    4. 🏋️‍♀️ மறுவாழ்வு வழிகாட்டுதல்
    5. 🌐 வளங்கள் & தொடர்புகள்
    6. ⚠️ அவசர பதில்

    மெனுவுக்குத் திரும்ப 'menu' அல்லது 'back' ஐ தட்டச்சு செய்யவும்
    பிரிவு குறிப்பிட்ட வழிகாட்டுதலுக்கு 'help' ஐ தட்டச்சு செய்யவும்"""
}

# Multilingual help messages for each section
section_help = {
    "1": {
        "en": """Stroke Education Help:
    - Learn about different types of strokes
    - Understand stroke risk factors
    - Get information about stroke prevention
    - Ask about post-stroke complications
    Example: "What are the early signs of stroke?"
    """,
        "zh": """中风教育帮助：
    - 了解不同类型的中风
    - 理解中风风险因素
    - 获取中风预防信息
    - 询问中风后并发症
    示例："中风的早期征兆是什么？"
    """,
        "ms": """Bantuan Pendidikan Strok:
    - Pelajari tentang jenis-jenis strok
    - Fahami faktor risiko strok
    - Dapatkan maklumat pencegahan strok
    - Tanya tentang komplikasi selepas strok
    Contoh: "Apakah tanda-tanda awal strok?"
    """,
            "ta": """பக்கவாதக் கல்வி உதவி:
    - வெவ்வேறு வகையான பக்கவாதங்களைப் பற்றி அறியவும்
    - பக்கவாத ஆபத்து காரணிகளை புரிந்துகொள்ளவும்
    - பக்கவாத தடுப்பு தகவல்களைப் பெறவும்
    - பக்கவாதத்திற்குப் பிந்தைய சிக்கல்களைப் பற்றி கேட்கவும்
    உதாரணம்: "பக்கவாதத்தின் ஆரம்ப அறிகுறிகள் என்ன?"
    """
        },
        "2": {
            "en": """Medication Management Help:
    - Get information about stroke medications
    - Learn about proper dosage and timing
    - Understand side effects
    - Get storage instructions
    Example: "What are common blood thinners for stroke?"
    """,
            "zh": """药物管理帮助：
    - 获取中风药物信息
    - 了解正确的剂量和时间
    - 理解副作用
    - 获取储存说明
    示例："中风常用的血液稀释剂有哪些？"
    """,
            "ms": """Bantuan Pengurusan Ubat:
    - Dapatkan maklumat tentang ubat-ubatan strok
    - Pelajari tentang dos dan masa yang betul
    - Fahami kesan sampingan
    - Dapatkan arahan penyimpanan
    Contoh: "Apakah pengcair darah yang biasa untuk strok?"
    """,
            "ta": """மருந்து மேலாண்மை உதவி:
    - பக்கவாத மருந்துகள் பற்றிய தகவல்களைப் பெறவும்
    - சரியான அளவு மற்றும் நேரம் பற்றி அறியவும்
    - பக்க விளைவுகளை புரிந்துகொள்ளவும்
    - சேமிப்பு வழிமுறைகளைப் பெறவும்
    உதாரணம்: "பக்கவாதத்திற்கான பொதுவான இரத்த மெல்லிதாக்கிகள் என்ன?"
    """
        },
        "3": {
            "en": """Emotional Support Help:
    - Get daily motivation
    - Learn stress management techniques
    - Find coping strategies
    - Access mental health resources
    Example: "How can I manage caregiver stress?"
    """,
            "zh": """情感支持帮助：
    - 获取每日激励
    - 学习压力管理技巧
    - 寻找应对策略
    - 获取心理健康资源
    示例："如何管理护理人员的压力？"
    """,
            "ms": """Bantuan Sokongan Emosi:
    - Dapatkan motivasi harian
    - Pelajari teknik pengurusan tekanan
    - Cari strategi menghadapi
    - Akses sumber kesihatan mental
    Contoh: "Bagaimana saya boleh menguruskan tekanan penjaga?"
    """,
            "ta": """உணர்ச்சி ஆதரவு உதவி:
    - தினசரி ஊக்கத்தைப் பெறவும்
    - மன அழுத்த மேலாண்மை நுட்பங்களை அறியவும்
    - சமாளிக்கும் உத்திகளைக் கண்டறியவும்
    - மன நல ஆதாரங்களை அணுகவும்
    உதாரணம்: "பராமரிப்பாளர் மன அழுத்தத்தை எவ்வாறு நிர்வகிப்பது?"
    """
        },
        "4": {
            "en": """Rehabilitation Guidance Help:
    - Get exercise tutorials
    - Learn about daily activity adaptations
    - Understand rehabilitation techniques
    - Track recovery progress
    Example: "What exercises help with arm strength?"
    """,
            "zh": """康复指导帮助：
    - 获取运动教程
    - 了解日常活动适应
    - 理解康复技巧
    - 跟踪恢复进展
    示例："哪些运动有助于增强手臂力量？"
    """,
            "ms": """Bantuan Panduan Pemulihan:
    - Dapatkan tutorial senaman
    - Pelajari tentang penyesuaian aktiviti harian
    - Fahami teknik pemulihan
    - Jejak kemajuan pemulihan
    Contoh: "Apakah senaman yang membantu kekuatan lengan?"
    """,
            "ta": """மறுவாழ்வு வழிகாட்டுதல் உதவி:
    - உடற்பயிற்சி பயிற்சிகளைப் பெறவும்
    - தினசரி செயல்பாட்டு தகவமைப்புகளைப் பற்றி அறியவும்
    - மறுவாழ்வு நுட்பங்களை புரிந்துகொள்ளவும்
    - மீட்பு முன்னேற்றத்தை கண்காணிக்கவும்
    உதாரணம்: "கை வலிமைக்கு எந்த பயிற்சிகள் உதவும்?"
    """
        },
        "5": {
            "en": """Resources & Contacts Help:
    - Find local stroke rehabilitation centers
    - Access financial assistance information
    - Get support helplines
    - Find transport and home care services
    - Access video resources and guides
    Example: "What stroke support services are available near me?"
    """,
            "zh": """资源与联系帮助：
    - 查找当地中风康复中心
    - 获取财务援助信息
    - 获取支持热线
    - 查找交通和家庭护理服务
    - 访问视频资源和指南
    示例："我附近有哪些中风支持服务？"
    """,
            "ms": """Bantuan Sumber & Hubungan:
    - Cari pusat pemulihan strok tempatan
    - Akses maklumat bantuan kewangan
    - Dapatkan talian bantuan sokongan
    - Cari perkhidmatan pengangkutan dan penjagaan di rumah
    - Akses sumber dan panduan video
    Contoh: "Apakah perkhidmatan sokongan strok yang tersedia berdekatan saya?"
    """,
            "ta": """வளங்கள் & தொடர்புகள் உதவி:
    - உள்ளூர் பக்கவாத மறுவாழ்வு மையங்களைக் கண்டறியவும்
    - நிதி உதவி தகவல்களை அணுகவும்
    - ஆதரவு உதவி எண்களைப் பெறவும்
    - போக்குவரத்து மற்றும் வீட்டு பராமரிப்பு சேவைகளைக் கண்டறியவும்
    - வீடியோ வளங்கள் மற்றும் வழிகாட்டிகளை அணுகவும்
    உதாரணம்: "என் அருகில் என்ன பக்கவாத ஆதரவு சேவைகள் உள்ளன?"
    """
        },
        "6": {
            "en": """Emergency Response Help:
    - Learn FAST stroke detection
    - Understand when to call emergency services
    - Get first aid guidance
    - Monitor vital signs
    Example: "What are the FAST signs of stroke?"
    """,
            "zh": """紧急响应帮助：
    - 学习FAST中风检测
    - 了解何时呼叫紧急服务
    - 获取急救指导
    - 监测生命体征
    示例："FAST中风征兆是什么？"
    """,
            "ms": """Bantuan Respons Kecemasan:
    - Pelajari pengesanan strok FAST
    - Fahami bila hendak menghubungi perkhidmatan kecemasan
    - Dapatkan panduan pertolongan cemas
    - Pantau tanda-tanda vital
    Contoh: "Apakah tanda-tanda FAST untuk strok?"
    """,
            "ta": """அவசர பதில் உதவி:
    - FAST பக்கவாத கண்டறிதலைக் கற்றுக்கொள்ளவும்
    - அவசர சேவைகளை எப்போது அழைக்க வேண்டும் என்பதை புரிந்துகொள்ளவும்
    - முதலுதவி வழிகாட்டுதலைப் பெறவும்
    - உயிர்நாடி அறிகுறிகளை கண்காணிக்கவும்
    உதாரணம்: "பக்கவாதத்தின் FAST அறிகுறிகள் என்ன?"
    """
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
        # Load PDFs
        for pdf_path in pdf_paths:
            try:
                loader = PyPDFLoader(pdf_path)
                try:
                    pdf_text = loader.load()
                    combined_text += "\n\n".join([doc.page_content for doc in pdf_text]) + "\n\n"
                except Exception as e:
                    print(f"Error loading {pdf_path}: {str(e)}")
                    continue
            except Exception as e:
                print(f"Error with PDF {pdf_path}: {str(e)}")
                continue

        # Add medication guide content directly from txt file
        try:
            with open("./data/medication_guide.txt", "r", encoding='utf-8') as f:
                medication_guide = f.read()
            combined_text += "\n\n" + medication_guide
        except Exception as e:
            print(f"Error loading medication guide: {str(e)}")

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

    def get_context_for_navigation(self, section_number):
        contexts = {
            "1": "Context: Information about stroke types, causes, risk factors, prevention, and warning signs.",
            "2": "Context: Information about stroke medications including antiplatelets, anticoagulants, blood pressure medications, and statins, their dosages, and usage guidelines.",
            "3": "Context: Support and coping strategies for stroke caregivers, including stress management and emotional wellbeing.",
            "4": "Context: Stroke rehabilitation exercises, physical therapy techniques, and recovery guidance.",
            "5": "Context: Focus ONLY on available resources including: video resources from CaknaStroke, local stroke support services, medical facilities, and financial assistance programs in Malaysia. When asked about videos or educational content, prioritize providing information about available video resources.",
            "6": "Context: Emergency response protocols, FAST method, and immediate actions for stroke emergencies."
        }
        return contexts.get(section_number, "")

    def extract_name_with_llm(self, user_input):
        extraction_prompt = f"Extract the name from this sentence: '{user_input}'. If there is no name, respond with 'None'. Keep in mind, just provide the name and nothing else."
        response = llm.invoke(extraction_prompt)
        extracted_name = response.content.strip()
        return None if extracted_name.lower() == "none" else extracted_name

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception as e:
            print(f"Language detection error: {e}")
            return "en"

    def process_query(self, query, section_number=None):
        direct_instruction = "Provide direct, clear answers without phrases like 'according to the provided information' or 'the information states'. Just state the facts directly."
        
        if section_number:
            context = self.get_context_for_navigation(section_number)
            
            # Special handling for Resources section
            if section_number == "5" and ("video" in query.lower() or "watch" in query.lower()):
                specific_instruction = "Focus ONLY on providing information about available video resources and educational content. Do not provide general stroke information."
                query_with_context = f"{direct_instruction}\n\n{specific_instruction}\n\n{context}\n\nUser query: {query}"
            else:
                query_with_context = f"{direct_instruction}\n\n{context}\n\nUser query: {query}"
                
            return self.qa_interface.invoke({"query": query_with_context})["result"]
        return self.qa_interface.invoke({"query": f"{direct_instruction}\n\nUser query: {query}"})["result"]

chatbot_agent = ChatbotAgent()

@app.route("/webhook", methods=['POST'])
def webhook():
    incoming_msg = request.values.get('Body', '').strip()
    phone_number = request.values.get('From', '')
    detected_language = chatbot_agent.detect_language(incoming_msg)
    
    # Initialize user data if not exists
    if phone_number not in user_data:
        user_data[phone_number] = {"name": None, "current_section": None}

    twilio_response = MessagingResponse()

    # Handle name collection
    if user_data[phone_number]["name"] is None:
        if incoming_msg.lower() in ["hi", "hey", "hello"]:
            name_request = {
                "en": "Hi there! Before we continue, could you please tell me your name?",
                "zh": "您好！在继续之前，您能告诉我您的名字吗？",
                "ms": "Hai! Sebelum kita teruskan, bolehkah anda beritahu saya nama anda?",
                "ta": "வணக்கம்! தொடருவதற்கு முன் உங்கள் பெயரை சொல்ல முடியுமா?"
            }
            twilio_response.message(name_request.get(detected_language, name_request["en"]))
            return str(twilio_response)
        else:
            extracted_name = chatbot_agent.extract_name_with_llm(incoming_msg)
            if extracted_name:
                user_data[phone_number]["name"] = extracted_name
                # Send welcome message and navigation menu
                welcome_message = {
                    "en": f"Thank you, {extracted_name}!",
                    "zh": f"谢谢您, {extracted_name}！",
                    "ms": f"Terima kasih, {extracted_name}!",
                    "ta": f"நன்றி, {extracted_name}!"
                }
                twilio_response.message(welcome_message.get(detected_language, welcome_message["en"]))
                twilio_response.message(navigation_menu.get(detected_language, navigation_menu["en"]))
                return str(twilio_response)
            else:
                name_request = {
                    "en": "I couldn't quite catch your name. Could you please provide it again?",
                    "zh": "我没能听清您的名字。请您再提供一次好吗？",
                    "ms": "Saya tidak dapat menangkap nama anda. Bolehkah anda berikan sekali lagi?",
                    "ta": "நான் உங்கள் பெயரைப் பிடிக்க முடியவில்லை. தயவுசெய்து மீண்டும் சொல்லுங்கள்."
                }
                twilio_response.message(name_request.get(detected_language, name_request["en"]))
                return str(twilio_response)

    # Handle navigation commands
    if incoming_msg.lower() in ["menu", "back"]:
        user_data[phone_number]["current_section"] = None
        twilio_response.message(navigation_menu.get(detected_language, navigation_menu["en"]))
        return str(twilio_response)

    # Handle help command
    if incoming_msg.lower() == "help":
        current_section = user_data[phone_number]["current_section"]
        if current_section:
            twilio_response.message(section_help[current_section].get(detected_language, section_help[current_section]["en"]))
        else:
            twilio_response.message(navigation_menu.get(detected_language, navigation_menu["en"]))
        return str(twilio_response)

    # Handle section selection
    if incoming_msg in ["1", "2", "3", "4", "5", "6"]:
        user_data[phone_number]["current_section"] = incoming_msg
        twilio_response.message(section_help[incoming_msg].get(detected_language, section_help[incoming_msg]["en"]))
        return str(twilio_response)

    # Process query based on current section
    current_section = user_data[phone_number]["current_section"]
    if current_section is None:
        twilio_response.message(navigation_menu.get(detected_language, navigation_menu["en"]))
        return str(twilio_response)

    # Get response using the chatbot agent
    response_text = chatbot_agent.process_query(incoming_msg, current_section)
    user_name = user_data[phone_number]["name"]
    formatted_response = f"{user_name}, {response_text}"

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
