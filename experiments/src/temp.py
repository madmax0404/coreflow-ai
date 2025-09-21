import asyncio
from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()

def run_query_convert_agent(query: str) -> str:
    instructions = """You are an assistant that rewrites user input into a concise search query.
- Focus on extracting the key entities, concepts, and intent.
- Remove unnecessary filler words.
- Make the query suitable for retrieving relevant passages from a knowledge base.

User input: "{user_prompt}"
Search query:"""
    qc_agent = Agent(name="Query Convert Assistant", instructions=instructions)
    rag_query_result = asyncio.run(Runner.run(qc_agent, f'User input: "{query}"\nSearch query:'))
    compressed = (rag_query_result.final_output or "").replace("Search query:", "").strip() or query
    
    return compressed

res = run_query_convert_agent("hi")

print(res)