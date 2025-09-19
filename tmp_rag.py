import asyncio
from dotenv import load_dotenv
load_dotenv()
from src.agents.mcp import server

async def main():
    res = await server.rag("CoreFlow 휴가 규정")
    print(res.type, len(res.text))

asyncio.run(main())
