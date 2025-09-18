import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SERVER_PATH = os.path.join(BASE_DIR, "agents\mcp\server.py")
SERVER_CWD  = os.path.dirname(SERVER_PATH)
PROJECT_ROOT = os.path.join(BASE_DIR, "..")

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
    
instructions = """사용자의 질문에 답변할 때, 필요한 도구들이 있다면 사용하여 답변하도록 하세요.
만약, 필요한 도구가 없고, 답변에 필요한 정보 또한 없다면, 솔직하게 모른다고 답변하세요."""