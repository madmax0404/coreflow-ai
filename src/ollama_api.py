import requests
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    model="gemma3:4b-it-qat"
    url = "http://100.103.21.230:11434/api/chat"
    
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