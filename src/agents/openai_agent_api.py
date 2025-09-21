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
    
@app.post("/chat")
async def chat_with_openai_agent(chat_request: ChatRequest):
    async with MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "python", SERVER_PATH, "stdio"],
        },
        client_session_timeout_seconds=120,
    ) as server:
    
        agent = Agent(
            name="CoreFlow AI",
            instructions=instructions,
            mcp_servers=[server],
        )
        
        result = await Runner.run(agent, chat_request.messages)
        
        return result.final_output

class TitleRequest(BaseModel):
    message: str

@app.post("/create_title")
async def create_title(title_request: TitleRequest):
    agent = Agent(
        name="Title Creator Assistant",
        instructions="사용자의 질문에 알맞는 20글자 미만의 짧은 제목을 한글로 작성해줘."
    )
    
    result = await Runner.run(agent, title_request.message)
    
    return result.final_output