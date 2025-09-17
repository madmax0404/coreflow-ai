from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os, requests, pandas as pd, httpx, asyncio
from langchain_chroma import Chroma
from agents import Agent, Runner
from typing import List, Optional
import nest_asyncio
nest_asyncio.apply()
from pydantic import BaseModel

load_dotenv()

headers = {
    "authorization": os.getenv("myserver_api_key")
}

API_KEY = os.getenv("myserver_api_key")

embed_url = os.getenv("myserver_url")+":8000/embed"
rerank_url = os.getenv("myserver_url")+":8000/rerank"

GEOCODE_API_URL = os.getenv("geocode_url")
geocode_api_key = os.getenv("geocode_api_key")
FORECAST_API_URL = os.getenv("forecast_api_url")
WEATHER_TIMEZONE = "Asia/Seoul"

WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

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

# 지역 이름을 입력하면 경도와 위도를 반환해주는 함수
def geocode_location(query: str) -> Optional[dict]:
    base_params = {"key": geocode_api_key, "address": query}
    
    geocode_response = requests.get(GEOCODE_API_URL, params=base_params)
    
    geocode_result = geocode_response.json()["results"][0]

    if geocode_result:
        return geocode_result["geometry"]["location"]

    return None


def fetch_current_weather(location: str) -> str:
    query = (location or "").strip()
    if not query:
        return "Please provide a location name."
    
    location_info = geocode_location(query)
    if not location_info:
        return f"Could not find coordinates for '{query}'."
    latitude = location_info["lat"]
    longitude = location_info["lng"]

    if latitude is None or longitude is None:
        return f"Incomplete coordinate data for '{query}'."

    try:
        weather_response = requests.get(
            FORECAST_API_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
                "timezone": WEATHER_TIMEZONE,
            },
        )
        weather_data = weather_response.json()

        current_weather = weather_data["current_weather"] or {}
        if not current_weather:
            return "Weather service returned an empty response."

    except httpx.HTTPStatusError as error:
        return f"Weather service error: {error.response.status_code}."
    except httpx.RequestError as error:
        return f"Unable to reach weather service: {error}."

    weather_code = current_weather["weathercode"]
    description = WEATHER_CODE_DESCRIPTIONS.get(weather_code, "Unknown conditions")
    temperature = current_weather.get("temperature")
    windspeed = current_weather.get("windspeed")
    observation_time = current_weather.get("time")

    temperature_text = f"{temperature:.1f}°C" if isinstance(temperature, (int, float)) else "N/A"
    wind_text = f"{windspeed:.1f} m/s" if isinstance(windspeed, (int, float)) else "N/A"

    return (
        f"Weather for {query}: {description}. "
        f"Temperature {temperature_text}, wind {wind_text}. "
        f"Observed at {observation_time or 'unknown time'} ({WEATHER_TIMEZONE})."
    )

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

# class RagReq(BaseModel):
#     query: str

@mcp.tool()
def rag(query:str) -> str:
    """CoreFlow 사내 문서 검색"""
    
    # query = rag_req.query
    
    if not parent_docs_list:
        return "RAG 데이터가 준비되지 않았습니다. 관리자에게 DATA_PATH를 확인하세요."
    
    compressed = asyncio.run(run_query_convert_agent(query))["out"]
    
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


@mcp.tool()
def current_weather(location: str) -> str:
    """Return the current weather for the requested location in Korea."""

    return fetch_current_weather(location)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
