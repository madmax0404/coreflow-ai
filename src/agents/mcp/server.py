from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv
import asyncio
import logging
import os, requests, pandas as pd, httpx, json, time
from langchain_chroma import Chroma
from agents import Agent, Runner
from typing import List, Optional, Any, Awaitable, TypeVar
import nest_asyncio
nest_asyncio.apply()
from mcp.types import TextContent

load_dotenv()

mcp = FastMCP("RAG and Weather MCP Server")

headers = {
    "authorization": os.getenv("myserver_api_key")
}

API_KEY = os.getenv("myserver_api_key")

embed_url = os.getenv("myserver_url")+":8001/embed"
rerank_url = os.getenv("myserver_url")+":8001/rerank"

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

REQUEST_TIMEOUT = 180


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
# print(DB_PATH)

# CSV/VectorStore는 프로세스 시작 시 1회 로드
parent_docs_list: List[str] = []
if os.path.isfile(DATA_PATH):
    parent_docs_df = pd.read_csv(DATA_PATH)
    parent_docs_list = parent_docs_df["docs"].dropna().astype(str).tolist()
else:
    # 파일 없으면 나중에 친절하게 에러 주도록 한다
    parent_docs_list = []

# RAG용 질문으로 변환
async def run_query_convert_agent(query: str) -> str:
    instructions = """You are an assistant that rewrites user input into a concise search query.
- Focus on extracting the key entities, concepts, and intent.
- Remove unnecessary filler words.
- Make the query suitable for retrieving relevant passages from a knowledge base.

User input: "{user_prompt}"
Search query:"""
    qc_agent = Agent(name="Query Convert Assistant", instructions=instructions)
    rag_query_result = await Runner.run(qc_agent, f'User input: "{query}"\nSearch query:')
    compressed = (rag_query_result.final_output or "").replace("Search query:", "").strip() or query
    
    return compressed

@mcp.tool()
async def rag(query: str) -> TextContent:
    """The company(CoreFlow) internal document search tool."""
    normalized_query = query if isinstance(query, str) else json.dumps(query, ensure_ascii=False)
    normalized_query = normalized_query.replace("CoreFlow", "")
    normalized_query = (normalized_query or "").strip()
    if not normalized_query:
        return TextContent(type="text", text="Please provide a query to search.", mimeType="text/plain")

    if not parent_docs_list:
        return TextContent(
            type="text",
            text="RAG index is not ready. Please verify DATA_PATH.",
            mimeType="text/plain",
        )

    try:
        compressed = await run_query_convert_agent(normalized_query)
        # print(compressed)
    except Exception as exc:
        return TextContent(
            type="text",
            text=f"Failed to prepare search query: {exc}",
            mimeType="text/plain",
        )

    embeddings = CustomEmbeddings()
    try:
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings,
        )
        retriever = vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50}
        )
        docs_list = retriever.invoke(compressed)
    except Exception as exc:
        return TextContent(
            type="text",
            text=f"Vector store error: {exc}",
            mimeType="text/plain",
        )

    first_searched_parent_docs = search_parent_docs(docs_list, parent_docs_list)
    searched_parent_docs_df = pd.DataFrame({"docs": first_searched_parent_docs}).drop_duplicates()
    searched_parent_docs = searched_parent_docs_df["docs"].tolist()

    if not searched_parent_docs:
        return TextContent(type="text", text="No related documents found.", mimeType="text/plain")

    try:
        relevant_docs = []
        scores = []
        for searched_parent_doc in searched_parent_docs:
            rerank_response = requests.post(rerank_url, json={"query":compressed, "documents":[searched_parent_doc]}, headers=headers)
            rerank_data = rerank_response.json()
            if rerank_data["results"][0]["score"] >= 0.9:
                relevant_docs.append(rerank_data["results"][0]["text"])
                scores.append(rerank_data["results"][0]["score"])
        # rerank_response = requests.post(rerank_url, json={"query":compressed, "documents":searched_parent_docs}, headers=headers)
        # rerank_data = rerank_response.json()
    except RuntimeError as exc:
        return TextContent(type="text", text=str(exc), mimeType="text/plain")
    
    if relevant_docs:
        relevant_docs_df = pd.DataFrame({"docs":relevant_docs, "scores":scores}).drop_duplicates().sort_values(by="scores", ascending=False)

        if len(relevant_docs) > 1:
            child_docs = []
            for doc1 in relevant_docs:
                for doc2 in relevant_docs:
                    if doc1 == doc2:
                        continue
                    else:
                        if doc1 in doc2:
                            child_docs.append(doc1)

            relevant_docs_df = relevant_docs_df[~relevant_docs_df["docs"].isin(child_docs)]
        relevant_docs_df = relevant_docs_df.head(5)
        # print(relevant_docs_df)
    else:
        return TextContent(type="text", text="No relevant documents found.", mimeType="text/plain")


    
    # print(rerank_data)

    # results = rerank_data.get("results") if isinstance(rerank_data, dict) else None

    # print(len(results))

    # for result in results:
    #     print(result)


    # best_results: list[str] = []
    # if isinstance(relevant_docs, list):
    #     for result in relevant_docs:
    #         text = result.get("text") if isinstance(result, dict) else None
    #         if text:
    #             best_results.append(text)
    #         if len(best_results) == 5:
    #             break

    best_results = relevant_docs_df["docs"].tolist()

    if not best_results:
        return TextContent(type="text", text="No related documents found.", mimeType="text/plain")
    
    # print(best_results)

    
    out = "\n\n---\n\n".join(best_results)

    return TextContent(type="text", text=out, mimeType="text/plain")
    # return out

# result = asyncio.run(rag("CoreFlow의 휴가 규정에 대해서 알려줘."))

# print(result)


@mcp.tool()
def current_weather(location: str, ctx: Context) -> str:
    """Return the current weather for the requested location in Korea."""

    asyncio.run(ctx.info(f"location: {location}"))
    
    if isinstance(location, dict):
        location = location["title"]
    else:
        location = str(location)

    return fetch_current_weather(location)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
