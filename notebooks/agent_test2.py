import os, pathlib, asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

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

tools = asyncio.run(client.get_tools())

prompt = hub.pull("hwchase17/structured-chat-agent")

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

history = agent_with_history.get_session_history("user-123")

print(history)

print(type(history))

async def run_agent():
    res = await agent_with_history.ainvoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
    print(res)
    
asyncio.run(run_agent())

# res = agent_with_history.invoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
# print(res)

