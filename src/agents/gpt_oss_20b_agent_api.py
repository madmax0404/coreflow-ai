import os, pathlib, asyncio, requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from datetime import timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.agents import AgentActionMessageLog  # Agent step type containing message logs

app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # 웹브라우저와 파이썬 서버간에 통신을 허용하기 위한 미들웨어
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

headers = {
    "authorization": os.getenv("myserver_api_key")
}

ollama_url = os.getenv("ollama_url")
gpt_model = os.getenv("gpt_model")

llm = ChatOllama(
    model=gpt_model,
    base_url=ollama_url,
    keep_alive=0,
    num_ctx=32000,
    reasoning=True,
    verbose=True,
)

BASE_DIR = pathlib.Path.cwd()
SERVER_PATH = (BASE_DIR / ".." / "src" / "agents" / "mcp" / "server.py").resolve()

client = MultiServerMCPClient(
    {
        "RAG_and_weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
            "timeout": timedelta(seconds=180)
        },
    }
)

# mcp_tools = asyncio.run(client.get_tools())

# async def create_tools():
#     return await client.get_tools()

# mcp_tools = create_tools()

system_prompt = """Reasoning: high
You are an assistant AI of a company called CoreFlow.
Use tools when necessasry.
Your final answer MUST BE FOCUSED ONLY on the topic of the user's question; If tool response give you extra information that is not directly related to the user's question, YOU MUST ignore them.
DO NOT provide any information that you don't actually have."""

class ChatRequest(BaseModel):
    messages: list
    
@app.post("/chat")
async def chat_with_ollama(chat_request: ChatRequest):
    
    messages = chat_request.messages
    
    # print(messages)
    # print(type(messages))
    # print(type(messages[0]))
    
    prompt_messages = []
    for message in messages:
        if message["role"] == "user":
            prompt_messages.append(("human", message["content"]))
        elif message["role"] == "assistant":
            prompt_messages.append(("ai", message["content"]))
    
    # print(prompt_messages[:-1])
    
    added_messages = [
        ("system", system_prompt)
    ] + prompt_messages[:-1] + [
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
    
    # print(added_messages)
    
    prompt = ChatPromptTemplate.from_messages(added_messages)
    
    # print(prompt)
    
    mcp_tools = await client.get_tools()
    
    agent = create_tool_calling_agent(llm=llm, tools=mcp_tools, prompt=prompt)

    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True, return_intermediate_steps=True)

    res = await agent_executor.ainvoke({"input": messages[-1]["content"]})
    
    # print(res["output"])
    
    return res["output"]#.replace("\n", "<br/>")
    
class TitleRequest(BaseModel):
    message: str

@app.post("/create_title")
def create_title(title_request: TitleRequest):
    model = os.getenv("ollama_model")
    url = os.getenv("ollama_url") + "/api/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "사용자의 질문에 알맞는 20글자 미만의 짧은 제목을 한글로 작성해줘."},
            {"role": "user", "content": title_request.message}
        ],
        "stream": False,
        "keep_alive": 0
    }
    
    response = requests.post(url, json=payload).json()
    
    return response
