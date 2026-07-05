# 🏥 MediCare — AI Powered Medical Assistant

> An AI-powered Medical Assistant built using **React, Node.js, FastAPI, MongoDB, LangChain, and Google Gemini** that provides medical guidance, sentiment-aware responses, conversational memory, and secure user authentication.

---

## 🌐 Live Links

### 🚀 Frontend
https://medicare-22-beta.vercel.app/

### 🔗 Backend API
https://medicare-backend-kjc0.onrender.com

### 🤖 AI Chatbot API
https://medicare-chatbot-qhd7.onrender.com

---

# 📖 Overview

MediCare is a full-stack AI-powered medical assistant that combines modern web technologies with Large Language Models to provide an intelligent healthcare experience.

The application allows users to:

- 🔐 Create secure accounts
- 💬 Chat with an AI Medical Assistant
- ❤️ Receive sentiment-aware responses
- 🧠 Maintain conversational context
- 📜 Store and manage chat history
- 🏥 Get health-related guidance powered by Google Gemini

> **Disclaimer:** This application is intended for educational and informational purposes only. It is **not** a replacement for professional medical advice or emergency healthcare.

---

# ✨ Features

## 👨‍⚕️ AI Medical Assistant

- Medical Question Answering
- Symptom Guidance
- Mental Health Support
- First Aid Suggestions
- Medication Information
- Emergency Detection
- Context-aware Responses

---

## 😊 Sentiment Analysis

- Detects user emotion using VADER Sentiment Analysis
- Provides empathetic responses
- Emotion-aware prompt engineering
- Maintains sentiment history

---

## 🔐 Authentication

- User Signup
- Secure Login
- JWT Authentication
- Password Hashing using bcrypt
- Protected Routes

---

## 💬 Chat Features

- Conversation Memory
- Chat History
- Save Conversations
- Delete Conversations
- Personalized Responses

---

## ☁️ Deployment

- Frontend deployed on **Vercel**
- Backend deployed on **Render**
- Chatbot deployed on **Render**
- MongoDB Atlas Database
- Automatic deployment from GitHub

---

# 🛠 Tech Stack

## Frontend

- React 19
- Vite
- React Router
- Axios
- React Toastify

---

## Backend

- Node.js
- Express.js
- MongoDB Atlas
- Mongoose
- JWT
- bcrypt

---

## AI Backend

- FastAPI
- LangChain
- Google Gemini 2.5 Flash
- VADER Sentiment
- Pydantic

---

## Database

- MongoDB Atlas

---

## Deployment

- Vercel
- Render
- GitHub

---

# 📂 Project Structure

```
MediCare
│
├── my-app/                # React Frontend
│
├── backend/               # Express Backend
│
├── chatbot/               # FastAPI Medical Chatbot
│
├── PROJECT_ARCHITECTURE.md
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Saransh-22/MediCare.git

cd MediCare
```

---

# 🔧 Environment Variables

## Backend

Create:

```
backend/.env
```

```env
MONGO_URI=your_mongodb_connection_string

JWT_SECRET=your_secret_key

CHATBOT_URL=http://localhost:8000
```

---

## Chatbot

Create:

```
chatbot/.env
```

```env
GOOGLE_API_KEY=your_google_api_key
```

---

## Frontend

Create:

```
my-app/.env
```

```env
VITE_API_URL=http://localhost:5000
```

---

# 🚀 Running Locally

## Backend

```bash
cd backend

npm install

npm run dev
```

---

## Chatbot

```bash
cd chatbot

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn app:app --reload
```

---

## Frontend

```bash
cd my-app

npm install

npm run dev
```

Open

```
http://localhost:5173
```

---

# 🌍 Deployment

| Service | Platform |
|----------|----------|
| Frontend | Vercel |
| Backend | Render |
| AI Chatbot | Render |
| Database | MongoDB Atlas |

---

# 🔗 API Endpoints

## Authentication

```
POST /api/auth/signup

POST /api/auth/login
```

---

## Chat

```
POST /api/chat

GET /api/chat/history

POST /api/chat/save

DELETE /api/chat/history/:id
```

---

## AI Chatbot

```
POST /chat

GET /

GET /sentiment/{user_id}
```

---

# 🧠 Architecture

```
                User
                  │
                  ▼
      React Frontend (Vercel)
                  │
                  ▼
       Express Backend (Render)
                  │
        ┌─────────┴──────────┐
        ▼                    ▼
 MongoDB Atlas        FastAPI Chatbot
                              │
                              ▼
                    LangChain + Gemini
```

---

# 🔒 Security

- JWT Authentication
- Password Hashing using bcrypt
- Environment Variables
- CORS Configuration
- Protected API Routes
- MongoDB Atlas Security

---

# 🚀 Future Enhancements

### AI Features

- Voice-enabled Medical Assistant
- Image-based Disease Detection
- Medical Report Analysis (PDF)
- Medicine Recommendation Engine
- Health Risk Prediction
- Personalized Health Dashboard

---

### User Features

- Appointment Booking
- Doctor Recommendation
- Prescription History
- Family Health Profiles
- Multi-language Support
- Dark / Light Theme

---

### Technical Improvements

- Redis for Conversation Memory
- Docker Containerization
- Kubernetes Deployment
- CI/CD Pipeline using GitHub Actions
- Unit & Integration Testing
- API Rate Limiting
- Role-Based Access Control (RBAC)
- Refresh Token Authentication
- Logging & Monitoring
- AI Response Streaming

---

# 📈 Future Scope

- Integration with wearable health devices
- Electronic Health Records (EHR)
- Hospital Management Integration
- AI-powered symptom prediction
- Telemedicine support
- Mobile application (Android & iOS)

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```
git checkout -b feature-name
```

3. Commit changes

```
git commit -m "Added feature"
```

4. Push

```
git push origin feature-name
```

5. Open a Pull Request

---

# 📄 License

This project is developed for **educational and research purposes**.

---

# 👨‍💻 Author

## Saransh Neema

AI • Full Stack • Machine Learning • NLP

GitHub:
https://github.com/Saransh-22

---

⭐ If you found this project useful, consider giving it a star!