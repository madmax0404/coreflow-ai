# CoreFlow AI - RAG

KH정보교육원 파이널 프로젝트 ERP 시스템 CoreFlow의 AI 파트.

https://github.com/YunSangsoo/COREFLOW_FRONTEND

https://github.com/YunSangsoo/COREFLOW_BACKEND

---

## 프로젝트 개요

본 프로젝트는 KH정보교육원에서 파이널 프로젝트인 ERP시스템 CoreFlow를 제작하며 제가 맡은 파트인 사내 전용 AI 챗봇 기능을 구현하며 진행되었습니다.

목표는 RAG(Retrieval-Augmented Generation)를 사용하여 챗봇이 사내 규정 관련 답변을 할 수 있도록 하는 것이었습니다.

RAG란 사용자의 질문과 연관된 문서들을 검색/수집/첨부하여 LLM 모델이 해당 문서들을 참고하여 사용자의 질문에 답변할 수 있도록 하는 기술이며, 장점은 아래와 같습니다.
- 모델 답변에 최신 정보 반영
- Hallucination 감소
- 회사 내부 규정같은 특정 도메인 지식에 대한 답변 가능
- 파인튜닝이 필요하지 않음
- 고효율 & 확장성: 모든 지식을 대형 모델을 파인튜닝하여 파라미터에 담는 대신 소형 모델 사용 가능
- 대형 모델 사용에 필요한 GPU 수급 비용 절감

베이스 LLM은 구글의 소형 모델인 [Gemma3-4b-it-qat](https://deepmind.google/models/gemma/gemma-3/)을 [Ollama](https://ollama.com/)를 활용해 사용하였고, 임베딩 모델은 [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B#evaluation),
리랭커 모델은 [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B#evaluation)를 사용하였습니다. 모든 모델들은 아래 사양의 컴퓨터로 로컬 환경에서 구동하였습니다.
- **CPU**: AMD Ryzen 5800X3D
- **GPU**: RTX 3090
- **RAM**: 128GB

---

## 기술 스택

- **언어:** Python
- **LLM/임베딩/리랭커 모델:** [Gemma3-4b-it-qat](https://deepmind.google/models/gemma/gemma-3/) / [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B#evaluation) / [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B#evaluation)
- **LLM 스택:** [Ollama](https://ollama.com/), LangChain, [Transformers](https://huggingface.co/docs/transformers/index), [Sentence Transformers](https://sbert.net/), Chroma
- **데이터 처리:** Pydantic, pandas
- **웹 프레임워크:** FastAPI
- **개발 환경:** Windows 10/11, Linux Ubuntu Desktop 24.04 LTS, VS Code, Jupyter Notebook

---

## 문제

ERP 시스템을 개발하며 사내 전용 AI 챗봇이 **회사와 관련된 정보**에 대해서 답변을 할 수 있게 하려고 하였습니다.

이는 일반적인 LLM API 사용만으로는 불가능하며, **RAG**의 필요성을 의미했습니다.

---

## 데이터셋

회사 내부 규정은 구글 Gemini의 Deep Research 기능을 사용해 A4용지 약 20장 분량의 문서들을 생성했습니다.

---

## 방법론 및 접근 방식

이 프로젝트에서는 단순히 사용자 쿼리를 그대로 Vector DB에서 검색하여 쿼리와 유사한 문서를 가져오거나 그 검색 결과만을 사용한 것이 아닌, 새로운 아이디어를 적용해보았습니다.

















