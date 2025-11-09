import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = api_key

sentiment_analyzer = SentimentIntensityAnalyzer()

app = FastAPI(title="Medical Assistant Chatbot API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

user_memory = {}

def analyze_sentiment(text: str) -> dict:
    """
    Analyzes the sentiment and emotion of the user's input.
    Returns emotion type and sentiment scores.
    """
    scores = sentiment_analyzer.polarity_scores(text)
    compound = scores['compound']
    
    emotion = "neutral"
    motivation = ""
    
    text_lower = text.lower()
    
    anger_keywords = ['angry', 'furious', 'mad', 'frustrated', 'annoyed', 'irritated', 'rage', 'outraged', 'pissed', 'upset']
    
    pain_keywords = ['pain', 'hurt', 'ache', 'suffering', 'agony']
    
    anxiety_keywords = ['anxious', 'worried', 'scared', 'panic', 'nervous', 'fear', 'stress', 'terrified']
    
    sad_keywords = ['sad', 'depressed', 'hopeless', 'lonely', 'crying', 'down', 'miserable', 'grief']
    
    if any(word in text_lower for word in anger_keywords):
        emotion = "frustrated"
        motivation = "I can see you're feeling frustrated. Your feelings are valid. Let's focus on finding solutions that can help you feel better. 🤝"
    
    elif any(word in text_lower for word in pain_keywords):
        emotion = "in pain"
        motivation = "I'm sorry you're experiencing pain. Your wellbeing matters, and I'm here to help you understand your symptoms better. �"
    
    elif any(word in text_lower for word in anxiety_keywords):
        emotion = "anxious"
        motivation = "I understand you're feeling worried. Take a deep breath - you're not alone in this. Let's work through this together, one step at a time. 🌟"
    
    elif any(word in text_lower for word in sad_keywords) or compound < -0.5:
        emotion = "sad"
        motivation = "I can sense you're going through a difficult time. Remember, it's okay to feel this way, and seeking help is a sign of strength. You're taking a positive step by reaching out. 💙"
    
    elif compound > 0.5:
        emotion = "positive"
        motivation = "It's wonderful to see your positive spirit! Let's make sure you stay healthy and well. 😊"
    elif compound > 0.1:
        emotion = "calm"
        motivation = "I'm glad to help you with your health concerns. Your proactive approach to health is commendable. ✨"
    else:
        emotion = "neutral"
        motivation = "I'm here to assist you with your medical questions. Feel free to share your concerns. 🩺"
    
    return {
        "emotion": emotion,
        "compound": compound,
        "positive": scores['pos'],
        "negative": scores['neg'],
        "neutral": scores['neu'],
        "motivation": motivation
    }

medical_system_template = """
You are a **Professional Medical Assistant AI** designed to provide helpful, accurate, and empathetic medical information.

STRICT GUIDELINES:
1. **ONLY answer medical and health-related questions** including:
   - Symptoms analysis and possible conditions
   - General health advice and wellness tips
   - Medication information (general knowledge)
   - Preventive care and healthy lifestyle
   - Mental health support and guidance
   - First aid recommendations
   - When to seek emergency care

2. **If asked about NON-MEDICAL topics**, respond STRICTLY with:
   "I apologize, but I can only assist with medical and health-related concerns. Please ask me about symptoms, health conditions, wellness, or medical advice."

3. **NEVER hallucinate or make up information**:
   - Only provide information based on established medical knowledge
   - If uncertain, clearly state: "I'm not completely certain about this. Please consult a healthcare professional for accurate diagnosis."
   - Always recommend consulting a doctor for serious symptoms or diagnosis

4. **IMPORTANT DISCLAIMERS**:
   - You are NOT a replacement for professional medical care
   - For emergencies (chest pain, difficulty breathing, severe bleeding, etc.), ALWAYS advise: "This sounds like an emergency. Please call emergency services (911) or visit the nearest emergency room immediately."
   - For serious symptoms, ALWAYS recommend: "Please consult with a healthcare provider for proper examination and diagnosis."

5. **Be Empathetic and Supportive**:
   - Current user emotion: {emotion}
   - Acknowledge their feelings appropriately
   - Provide emotional support alongside medical information
   - Use a caring, professional tone

6. **Response Format - BE RELEVANT AND FOCUSED**:
   - Provide all RELEVANT information needed to answer the question
   - Avoid unnecessary background information or explanations
   - Skip lengthy introductions - get to the point quickly
   - Don't explain basic concepts unless specifically asked
   - Avoid repetitive statements or over-explaining
   - Use bullet points when listing multiple items (symptoms, tips, etc.)
   - Be clear and easy to understand
   - Explain medical terms if used, but keep explanations brief
   - Provide actionable advice when safe to do so
   - Focus on what the user needs to know, not everything you could say

EMOTIONAL CONTEXT:
User's current emotional state: {emotion}
Motivational message: {motivation}

CONVERSATION HISTORY:
{chat_history}

USER'S MEDICAL CONCERN:
{user_input}

ASSISTANT'S CONCISE RESPONSE:
"""

def unified_chat(user_input: str, user_id: str = None, detail_mode: str = "concise"):
    try:
        sentiment_data = analyze_sentiment(user_input)
        
        if user_id not in user_memory:
            user_memory[user_id] = {
                "history": ChatMessageHistory(),
                "sentiment_history": []
            }
        
        memory_data = user_memory[user_id]
        memory = memory_data["history"]
        
        memory_data["sentiment_history"].append({
            "emotion": sentiment_data["emotion"],
            "compound": sentiment_data["compound"]
        })
        
        chat_history = ""
        for msg in memory.messages[-10:]:  
            role = "User" if msg.type == "human" else "Assistant"
            chat_history += f"{role}: {msg.content}\n"
    
        prompt = ChatPromptTemplate.from_template(medical_system_template)
        formatted_prompt = prompt.format(
            chat_history=chat_history, 
            user_input=user_input,
            emotion=sentiment_data["emotion"],
            motivation=sentiment_data["motivation"]
        )
        
        response = llm.invoke(formatted_prompt)
        
        memory.add_user_message(user_input)
        memory.add_ai_message(response.content)
        
        return {
            "reply": response.content.strip(),
            "sentiment": sentiment_data
        }

    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return {
                "reply": "⚠️ Google Gemini API quota exceeded. Please check your API key or wait a minute.",
                "sentiment": {"emotion": "neutral", "compound": 0}
            }
        return {
            "reply": f"❌ Error: {str(e)}",
            "sentiment": {"emotion": "neutral", "compound": 0}
        }

class ChatRequest(BaseModel):
    user_input: str
    user_id: str | None = None
    detail_mode: str = "concise"

@app.post("/chat")
def chat(req: ChatRequest):
    """
    Medical Assistant Chatbot Endpoint
    Handles medical queries with sentiment analysis and emotional support.
    Returns both medical advice and sentiment analysis.
    """
    if not req.user_input.strip():
        raise HTTPException(status_code=400, detail="User input cannot be empty")

    result = unified_chat(req.user_input, req.user_id, req.detail_mode)
    return result

@app.get("/")
def root():
    return {
        "status": "Medical Assistant Chatbot API running 🏥",
        "features": [
            "Medical Q&A",
            "Sentiment Analysis",
            "Emotional Support",
            "Context-aware responses"
        ]
    }

@app.get("/sentiment/{user_id}")
def get_sentiment_history(user_id: str):
    """
    Get sentiment history for a specific user
    """
    if user_id not in user_memory:
        return {"sentiment_history": []}
    
    return {"sentiment_history": user_memory[user_id]["sentiment_history"]}
