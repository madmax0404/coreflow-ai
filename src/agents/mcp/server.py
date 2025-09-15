from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os, requests, pandas as pd, httpx, asyncio
from langchain_chroma import Chroma
from agents import Agent, Runner
from typing import List, Any
import nest_asyncio
nest_asyncio.apply()

load_dotenv()

headers = {
    "authorization": os.getenv("myserver_api_key")
}

API_KEY = os.getenv("myserver_api_key")

embed_url = os.getenv("myserver_url")+":8000/embed"
rerank_url = os.getenv("myserver_url")+":8000/rerank"

class CustomEmbeddings:
    def embed_documents(self, texts):
        embed_docs_payload = {
            "texts": texts,
            "is_query": False
        }
        return requests.post(embed_url, json=embed_docs_payload, headers=headers).json()["embeddings"]
    
    def embed_query(self, text):
        embed_query_payload = {
            "texts": [text],
            "is_query": True
        }
        return requests.post(embed_url, json=embed_query_payload, headers=headers).json()["embeddings"][0]

def search_parent_docs(docs, parent_docs):
    searched_docs = []
    for doc in docs:
        doc_content = doc.page_content
        for parent_doc in parent_docs:
            if doc_content in parent_doc:
                searched_docs.append(parent_doc)
    
    return searched_docs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # server.py 위치
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "data", "parent_split_docs_20250909.csv")
DB_PATH = os.path.join(BASE_DIR, "..", "..", "..", "chroma_db")

# CSV/VectorStore는 프로세스 시작 시 1회 로드
parent_docs_list: List[str] = []
if os.path.isfile(DATA_PATH):
    parent_docs_df = pd.read_csv(DATA_PATH)
    parent_docs_list = parent_docs_df["docs"].dropna().astype(str).tolist()
else:
    # 파일 없으면 나중에 친절하게 에러 주도록 한다
    parent_docs_list = []

# RAG용 질문으로 변환
async def run_query_convert_agent(query:str) -> str:
    instructions = """You are an assistant that rewrites user input into a concise search query.
- Focus on extracting the key entities, concepts, and intent.
- Remove unnecessary filler words.
- Make the query suitable for retrieving relevant passages from a knowledge base.

User input: "{user_prompt}"
Search query:"""
    qc_agent = Agent(name="Query Convert Assistant", instructions=instructions)
    rag_query_result = await Runner.run(qc_agent, f'User input: "{query}"\nSearch query:')
    compressed = (rag_query_result.final_output or "").replace("Search query:", "").strip() or query
    
    return {"out": compressed}
    

mcp = FastMCP("Local MCP Server for tools")

@mcp.tool()
def rag(query:str) -> str:
    """CoreFlow 사내 문서 검색"""
    
    if not parent_docs_list:
        return "RAG 데이터가 준비되지 않았습니다. 관리자에게 DATA_PATH를 확인하세요."
    
    compressed = asyncio.run(run_query_convert_agent(query))["out"]
    
    print(compressed)
    
    embeddings = CustomEmbeddings()
    
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})
    docs_list = retriever.invoke(compressed)
    
    first_searched_parent_docs = search_parent_docs(docs_list, parent_docs_list)
    searched_parent_docs_df = pd.DataFrame({"docs":first_searched_parent_docs})
    searched_parent_docs_df = searched_parent_docs_df.drop_duplicates()
    searched_parent_docs = searched_parent_docs_df["docs"].tolist()
    
    rerank_response = requests.post(rerank_url, json={"query":query, "documents":searched_parent_docs}, headers=headers)
    
    best_results = []
    for result in rerank_response.json()["results"][:5]:
        best_results.append(result["text"])
    
    if not best_results:
        return "관련 문서를 찾지 못했습니다."
    
    return "\n\n---\n\n".join(best_results)



if __name__ == "__main__":
    mcp.run()   # CLI 인자로 `stdio` 받으면 stdio로 기동