import pandas as pd
import requests
import json
import os
from langchain_chroma import Chroma
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import time
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,  # 웹브라우저와 파이썬 서버간에 통신을 허용하기 위한 미들웨어
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

headers = {
    "authorization": os.getenv("myserver_api_key")
}

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
    
embeddings = CustomEmbeddings()

vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

# mmr 기반
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":10, "fetch_k":50}
)

class ChatRequest(BaseModel):
    messages: list
    
def search_parent_docs(docs, parent_docs):
    searched_docs = []
    for doc in docs:
        doc_content = doc.page_content
        for parent_doc in parent_docs:
            if doc_content in parent_doc:
                searched_docs.append(parent_doc)
    
    return searched_docs

parent_docs_df = pd.read_csv("../data/parent_split_docs_20250909.csv")
parent_docs_list = parent_docs_df["docs"].tolist()

prompt_classification_prompt = """다음 사용자의 메시지를 분류하세요. 그 외의 메세지는 절대로 출력하지 마세요.
라벨 중 하나: chitchat, knowledge_rag
메시지: "{USER_MSG}"
출력: {"label": "...", "confidence": 0~1}"""

@app.post("/chat")
def chat_with_ollama(chat_request: ChatRequest):
    model = os.getenv("ollama_model")
    url = os.getenv("ollama_url") + "/chat"
    
    classification_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt_classification_prompt},
            chat_request.messages[-1]
        ],
        "stream": False,
        "keep_alive": 0
    }
    
    classification = requests.post(url, json=classification_payload).json()["message"]["content"]
    print(classification)
    
    if "knowledge_rag" not in classification:
        messages = chat_request.messages
    else:
        # RAG용 질문으로 변환
        rag_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": """You are an assistant that rewrites user input into a concise search query.
- Focus on extracting the key entities, concepts, and intent.
- Remove unnecessary filler words.
- Make the query suitable for retrieving relevant passages from a knowledge base.

User input: "{user_prompt}"
Search query:"""},
                chat_request.messages[-1]
            ],
            "stream": False,
            "keep_alive": 0
        }
        
        # RAG
        rag_query = requests.post(url, json=rag_payload).json()["message"]["content"].replace("Search query:", "").replace("\n", "").strip()
        print(rag_query)
        docs_list = retriever.get_relevant_documents(rag_query)
        
        # print(docs_list)
        
        searched_parent_docs = search_parent_docs(docs_list, parent_docs_list)
        searched_parent_docs_df = pd.DataFrame({"docs":searched_parent_docs})
        searched_parent_docs_df = searched_parent_docs_df.drop_duplicates()
        searched_parent_docs = searched_parent_docs_df["docs"].tolist()
        print(len(searched_parent_docs))
        
        # time.sleep(120)
        
        # Reranker
        rag_response = requests.post(rerank_url, json={"query":rag_query, "documents":searched_parent_docs}, headers=headers)
        
        # print(rag_response)
        
        best_results = []
        for result in rag_response.json()["results"][:5]:
            best_results.append(result["text"])
            
        messages = chat_request.messages[:-1]
        messages.append({"role": "system", "content": """Use the following context from the knowledge base to answer the question.
If the context is insufficient, say you don't know. Always answer in Korean.

Context:
{retrieved_docs}

Question:
{user_prompt}

Answer:"""})
        messages.append({"role": "user", "content": f"""
Context:
{best_results}

Question:
{chat_request.messages[-1]["content"]}

Answer:"""})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": 0,
        "options": {
            "num_ctx": 32000
        }
    }
    
    response = requests.post(url, json=payload).json()
    
    return response

class TitleRequest(BaseModel):
    message: str

@app.post("/create_title")
def create_title(title_request: TitleRequest):
    model = os.getenv("ollama_model")
    url = os.getenv("ollama_url") + "/chat"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "사용자의 질문에 알맞는 20글자 미만의 짧은 제목을 한글로 작성해줘."},
            {"role": "user", "content": title_request.message}
        ],
        "stream": False,
        "keep_alive": 0
    }
    
    response = requests.post(url, json=payload).json()
    
    return response