from pathlib import Path
from agents import Agent, Runner, OpenAIChatCompletionsModel, HostedMCPTool
from agents.mcp import MCPServerStdio
import asyncio
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model = OpenAI(
    base_url = os.getenv("ollama_url") + "v1",
    api_key='ollama', # required, but unused
)

my_model = OpenAIChatCompletionsModel(os.getenv("ollama_model"), openai_client=model)

print(os.listdir("./src/agents/mcp"))
# 샘플 파일의 경로 정의
samples_dir = "./src/agents/mcp"

async def main():
    # 비동기 컨텍스트 관리자를 사용하여 서버 초기화
    async with MCPServerStdio(
        params={
            "command": "uv",
            "args": ["run", "python", "./src/agents/mcp/test_server.py", "stdio"],
        }
    ) as server:
        # MCP 서버에서 제공하는 도구 목록 가져오기
        tools = await server.list_tools()
        print("[TOOLS]", [t.name for t in tools])

        # MCP 서버를 사용하는 에이전트 생성
        agent = Agent(
            name="Assistant",
            instructions="파일 시스템 도구를 사용하여 사용자의 작업을 도와주세요.",
            mcp_servers=[server],
            # model=my_model
        )

        # 에이전트 실행
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": "넌 어떤 도구를 사용할지 어떻게 판단해? 각 도구들의 description을 바탕으로 판단해?"},
            {"role": "assistant", "content": "네, 맞아요! 저는 제공된 **도구들의 description(설명)**을 바탕으로 어떤 도구를 사용할지 판단합니다."},
            {"role": "user", "content": "다시 한번 말해줄래?"}
        ]
        result = await Runner.run(agent, messages)
        print(result)
        print(result.final_output)
        
if __name__ == "__main__":
    asyncio.run(main())