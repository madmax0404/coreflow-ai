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
    messages: list

@app.post("/chat")
def chat_with_ollama(chat_request: ChatRequest):
    model = os.getenv("ollama_model")
    url = os.getenv("ollama_url") + "/chat"
    
    payload = {
        "model": model,
        "messages": chat_request.messages,
        "stream": False
    }
    
    response = requests.post(url, json=payload).json()
    
    return response

class TitleRequest(BaseModel):
    message: str

@app.post("/create_title")
def create_title(title_request: TitleRequest):
    model = os.getenv("ollama_model")
    url = os.getenv("ollama_url") + "/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "사용자의 질문에 알맞는 20글자 미만의 짧은 제목을 한글로 작성해줘."},
            {"role": "user", "content": title_request.message}
        ],
        "stream": False
    }
    
    response = requests.post(url, json=payload).json()
    
    return response