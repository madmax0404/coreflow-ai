import os, pathlib, asyncio, re
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

def _int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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
    # Force 32k context in Ollama and expire sessions immediately.
    extra_body={"keep_alive": 0, "options": {"num_ctx": 32000}},
)

BASE_DIR = pathlib.Path.cwd()
SERVER_PATH = (BASE_DIR / ".." / "src" / "agents" / "mcp" / "server.py").resolve()

MAX_RAG_SEGMENTS = max(1, _int_env("RAG_MAX_SEGMENTS", 6))
MAX_RAG_CHARS = max(200, _int_env("RAG_MAX_CHARS", 2000))

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
    if compiled:
        # Prefer sections that mention words from the user's query when possible.
        tokens = [t for t in re.split(r"\W+", query) if t]
        segments = [seg.strip() for seg in compiled.split("\n\n") if seg.strip()]
        selected = segments
        # if tokens:
        #     query_casefold = [t.casefold() for t in tokens]
        #     scored = []
        #     for seg in segments:
        #         seg_cf = seg.casefold()
        #         match_count = sum(1 for tok in query_casefold if tok and tok in seg_cf)
        #         if match_count:
        #             scored.append((match_count, -len(seg), seg))
        #     if scored:
        #         scored.sort(reverse=True)
        #         selected = [item[2] for item in scored[:MAX_RAG_SEGMENTS]]
        #     else:
        #         selected = segments[:MAX_RAG_SEGMENTS]
        # else:
        #     selected = segments[:MAX_RAG_SEGMENTS]

        compiled = "\n\n".join(selected)
        # if len(compiled) > MAX_RAG_CHARS:
        #     compiled = compiled[:MAX_RAG_CHARS]
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
    "- For 'rag', the only valid input is: {{\"query\": \"<text>\"}}.\n"
    "- Final response must be {{\"action\": \"Final Answer\", \"action_input\": \"<text>\"}}.\n"
    "- The final text is a single line containing 2-3 short facts about the user's request, separated by '; '.\n"
    "- Focus strictly on information relevant to the question (ignore unrelated policy sections).\n"
    "- If tool output mixes topics, prefer details that reuse the user's query terms.\n"
    "- Avoid double quotes inside the final text; rely on single quotes or paraphrase them.\n"
    "- Encode any newline as \\n if you must mention one, otherwise keep the text to a single line.\n"
)
prompt = ChatPromptTemplate.from_messages(
    [base_prompt.messages[0], ("system", strict_rules), *base_prompt.messages[1:]]
)

agent = create_structured_chat_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=(
        "FORMAT ERROR: respond with JSON like {\"action\": \"Final Answer\", \"action_input\": \"fact1; fact2\"}. "
        "Keep it to a single line with '; ' separators and avoid double quotes."
    ),
)

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
    print(res)
    
asyncio.run(run_agent())

# res = agent_with_history.ainvoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
# print(res)

# res = agent_with_history.invoke({"input": "CoreFlow의 휴가 규정에 대해서 알려줘."}, config=cfg)
# print(res)
