import asyncio
from dotenv import load_dotenv
load_dotenv()
from src.agents.mcp import server

async def main():
    for i in range(5):
        try:
            res = await server.rag("CoreFlow 휴가 규정")
            print(i, "len", len(res.text))
        except Exception as exc:
            print("error", i, exc)

asyncio.run(main())
