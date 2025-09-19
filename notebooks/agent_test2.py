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
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

ollama_url = os.getenv("ollama_url")
ollama_model = os.getenv("ollama_model")

if not ollama_url or not ollama_model:
    raise RuntimeError("ollama_url and ollama_model environment variables must be set.")

ollama_base_url = ollama_url.rstrip("/") + "/"

llm = ChatOpenAI(
    model=ollama_model,
    base_url=f"{ollama_base_url}v1",
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

try:
    tools_from_mcp = asyncio.run(client.get_tools())
except Exception as exc:
    raise RuntimeError("Failed to load tools from MCP server.") from exc

# ----------------------------
# 2) 원본 MCP 'rag' 툴 찾기 (프리픽스 유연 대응)
# ----------------------------
def _is_rag_tool(t):
    name = getattr(t, "name", "")
    return name == "rag" or name.endswith(":rag") or "rag" == name.split(":")[-1]

try:
    base_rag_tool = next(t for t in tools_from_mcp if _is_rag_tool(t))
except StopIteration:
    raise RuntimeError("MCP 서버에서 'rag' 툴을 찾지 못했습니다. 툴 이름/프리픽스를 확인하세요.")

# ----------------------------
# 3) StructuredTool 래퍼로 스키마 강제
# ----------------------------
class RagArgs(BaseModel):
    query: str = Field(..., description="Korean search query text")
    
async def _rag_call(query: str) -> str:
    """Wrap the underlying MCP rag tool for LangChain."""
    try:
        raw_result = await base_rag_tool.ainvoke({"query": query})
    except Exception as exc:
        return f"RAG tool error: {exc}"
    if isinstance(raw_result, list):
        compiled = '\n\n---\n\n'.join(
            str(item).strip() for item in raw_result if str(item).strip()
        )
    else:
        compiled = str(raw_result or "").strip()
    return compiled or "No related documents found."


rag_wrapped = StructuredTool.from_function(
    name="rag",
    description="Search internal CoreFlow documents and return top snippets.",
    args_schema=RagArgs,
    coroutine=_rag_call,  # 비동기 호출
)

# 원본 rag는 목록에서 제거하고, 대신 래퍼를 넣는다(중복 피함)
other_tools = [t for t in tools_from_mcp if t is not base_rag_tool]
tools = [rag_wrapped, *other_tools]

# ----------------------------
# 4) 프롬프트: structured-chat 프롬프트 + 포맷 수위 더 올리기(코드블록 금지)
# ----------------------------
base_prompt = hub.pull("hwchase17/structured-chat-agent")
from langchain_core.prompts import ChatPromptTemplate

strict_rules = (
    "IMPORTANT:\n"
    "- Do NOT use Markdown code fences.\n"
    "- When using a tool, output MUST be a JSON object matching the tool schema.\n"
    "- For 'rag', the only valid input is: {{\"query\": \"<text>\"}}\n"
)
prompt = ChatPromptTemplate.from_messages(
    [base_prompt.messages[0], ("system", strict_rules), *base_prompt.messages[1:]]
)

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

# res = agent_with_history.invoke({"input": "what's 1+1?"}, config=cfg)
# print(res)

# res = agent_with_history.invoke({"input": "what's the answer?"}, config=cfg)
# print(res)

# history = agent_with_history.get_session_history("user-123")

# print(history)

# print(type(history))

async def run_agent():
    res = await agent_with_history.ainvoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
    # print(res)
    
asyncio.run(run_agent())

# res = agent_with_history.ainvoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
# print(res)

# res = agent_with_history.invoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
# print(res)

