import os, pathlib, asyncio
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from datetime import timedelta

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

mcp_tools = asyncio.run(client.get_tools())

system_prompt = """Reasoning: high
You are an assistant AI of a company called CoreFlow.
Use tools when necessasry.
Your final answer MUST BE FOCUSED ONLY on the topic of the user's question; If tool response give you extra information that is not directly related to the user's question, YOU MUST ignore them.
DO NOT provide any information that you don't actually have."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    # ("human", "hi! my name is bob"),
    # ("ai", "Hello Bob! How can I assist you today?"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm=llm, tools=mcp_tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True, return_intermediate_steps=True)

res = asyncio.run(agent_executor.ainvoke({"input":"CoreFlow의 휴가 규정에 대해서 알려줘."}))

# print(res)
