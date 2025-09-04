import requests
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # 웹브라우저와 파이썬 서버간에 통신을 허용하기 위한 미들웨어
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_with_ollama(chat_request: ChatRequest):
    model = os.getenv("ollama_model")
    url = os.getenv("ollama_url") + "/chat"
    
    print(chat_request.question)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": chat_request.question}
        ],
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    return response.json()["message"]["content"]