import os, pathlib, uuid, sys, asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.tools import tool
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_mcp_adapters.tools import load_mcp_tools

ollama_url = os.getenv("ollama_url")
ollama_model = os.getenv("ollama_model")

llm = ChatOpenAI(
    model=ollama_model,
    base_url=ollama_url+"v1",
    api_key="ollama",
    temperature=0.2,
)

BASE_DIR = pathlib.Path.cwd()
SERVER_PATH = (BASE_DIR / ".." / "src" / "agents" / "mcp" / "server.py").resolve()

from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "RAG_and_weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
        },
    }
)

# print(client.session("RAG_and_weather"))

# async def get_tools():
#     async with client.session("RAG_and_weather") as session:
#         tools = await load_mcp_tools(session)
        
#         return tools
    
# tools = asyncio.run(get_tools())

# print(tools)

# tools = load_mcp_tools

tools = asyncio.run(client.get_tools())

# print(tools)

# prompt = hub.pull("hwchase17/react-chat")
prompt = hub.pull("hwchase17/structured-chat-agent")

# agent = create_react_agent(llm, tools, prompt=prompt)
agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

cfg = {"configurable": {"session_id": "user-123"}}

res = agent_with_history.invoke({"input": "what's 1+1?"}, config=cfg)
print(res)

res = agent_with_history.invoke({"input": "what's the answer?"}, config=cfg)
print(res)