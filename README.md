# CoreFlow AI - RAG

KH정보교육원 파이널 프로젝트 ERP 시스템 CoreFlow의 AI 파트.

https://github.com/YunSangsoo/COREFLOW_FRONTEND

https://github.com/YunSangsoo/COREFLOW_BACKEND

---

## 프로젝트 개요

본 프로젝트는 KH정보교육원에서 파이널 프로젝트인 ERP시스템 CoreFlow를 제작하며 제가 맡은 파트인 사내 전용 AI 챗봇 기능을 구현하며 진행되었습니다.

목표는 RAG(Retrieval-Augmented Generation)를 사용하여 챗봇이 **사내 규정** 관련 답변을 할 수 있도록 하는 것이었습니다.

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

![chat screenshot](<images/coreflowaichatexample.png>)
*Figure 1. CoreFlow AI 채팅 스크린샷*

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

ERP 시스템을 개발하며 사내 전용 AI 챗봇이 사내 규정 관련된 정보에 대해서 답변을 할 수 있게 하려고 하였습니다.

이는 일반적인 LLM API 사용만으로는 불가능하며, **RAG**의 필요성을 의미했습니다.

---

## 데이터셋

사내 규정은 구글 Gemini의 Deep Research 기능을 사용해 A4용지 약 20장 분량의 문서들을 생성했으며, 총 1장~11장으로 이루어져 있습니다.

---

## 방법론 및 접근 방식

이 프로젝트에서는 단순히 문서를 일정 chunk size로 쪼개서 Vector DB에 저장하고, 사용자 쿼리를 그대로 검색하여 유사한 문서를 가져오고, 그 검색 결과만을 사용하는 일반적인 RAG 파이프라인이 아닌, 새로운 아이디어를 적용해보았습니다.

1. **Splitting**
    - **대분류 - 유사도 검색용 문서**: 먼저, 모든 문서들을 **문장별**로 쪼개어 Vector DB에 저장하였습니다.
    - **대분류 - 부모 문서**: 그리고, 그 문장들이 포함될 **부모 문서들**을 각 장별, 소제목별, 문단별로 쪼개어 모두 CSV파일에 저장하였습니다.
2. **검색**
    - **Vector DB 검색용 쿼리로 변환**: 먼저, LLM을 이용하여 사용자의 질문을 Vector DB 검색에 최적화된 문장으로 변환합니다. 예) "CoreFlow의 휴가 규정에 대해서 알려줘." -> "CoreFlow 휴가 규정"
    - **Vector DB 검색**: 변환되어 최적화된 문장을 사용하여 MMR방식("k":10, "fetch_k":50)으로 유사한 문장들을 가져옵니다.
    - **부모 문서들 검색**: 위의 유사한 문장들이 포함된 부모 문서들을 검색하여 가져오고, 중복은 제거합니다.
    - **리랭커**: 원래의 사용자 질문과 위의 부모 문서들을 리랭커가 평가합니다.
3. **첨부**
    - 리랭커에 의해 평가된 부모 문서들 중 상위 5개를 첨부하여 사용자의 질문과 함께 모델에게 전달합니다.

Figure 1에서 확인 가능하듯, 이러한 방식은 성공적으로 모델이 사내 규정에 대해 대답할 수 있게 하였습니다.

추가적으로, RAG 파이프라인은 질문에 CoreFlow라는 단어가 포함되어 있을 때만 실행시켜, 잡담 또한 가능하게 하였습니다.

---

## 결과 및 주요 관찰

- **기존 chunk splitting 방식 대비 향상된 답변 품질**: 단순 chunk 기반 검색보다, 문장 단위 + 부모 문서 + 리랭커 구조를 적용했을 때 더 정밀하고 맥락 있는 답변을 제공.
- **Hallucination 감소**: 불필요하거나 근거 없는 답변 빈도가 줄었으며, 사내 규정 관련 질문에서 실제 문서 기반의 응답 비율이 높아짐.
- **검색 정확도 향상**: MMR 기반 검색 + 리랭커 조합으로, 사용자의 질문 의도를 더 잘 반영하는 문서들이 상위로 노출됨.
- **실용성 확보**: “CoreFlow” 키워드 필터링을 적용해, 챗봇이 규정 질의 응답과 잡담을 동시에 처리 가능.
- **로컬 환경에서 대규모 모델 운용 검증**: RTX 3090 기반 로컬 환경에서 Gemma3 + Qwen 시리즈 모델을 결합하여 실질적인 RAG 파이프라인 구축 가능성을 검증.

---

## 결론 및 향후 과제

본 프로젝트에서는 RAG(Retrieval-Augmented Generation) 기법을 활용하여 ERP 시스템 내 사내 규정 전용 챗봇을 성공적으로 구현하였습니다. 기존의 단순 chunk splitting 기반 접근 방식보다 향상된 검색 전략(문장 단위 분리 + 부모 문서 매핑 + 리랭킹)을 적용하여, 보다 정확하고 맥락 있는 답변을 제공할 수 있음을 확인하였습니다. 또한 RTX 3090 기반 로컬 환경에서 Gemma3 및 Qwen 시리즈 모델을 조합하여 실질적인 RAG 파이프라인 구축 가능성을 검증하였습니다.

향후 과제로는 다음과 같은 방향을 고려할 수 있습니다:
- **대규모 실제 문서 적용**: 시뮬레이션된 규정 문서가 아닌, 실제 기업의 방대한 규정·매뉴얼·지식 문서를 대상으로 검증 필요.
- **멀티모달 확장**: 텍스트 외에도 이미지/표 등 다양한 포맷을 처리하는 RAG로 확장.
- **모델 최적화 및 경량화**: 로컬 환경에서 더 빠른 추론을 위해 양자화, distillation, 캐싱 전략 적용.
- **사용자 경험 개선**: 챗봇 UI 개선, 질의 의도 파악(예: 요약, FAQ, 검색 모드 자동 전환) 기능 추가.
- **보안 및 접근 제어**: 사내 전용 데이터 사용 시 권한 관리와 로그 관리 기능 강화.
- **고사양 환경에서의 검증**: 현재는 RTX 3090으로 RAG 파이프라인을 구동했으나, VRAM이 눈물겨운 비명을 지르는 걸 확인했습니다. 향후에는 A100/H100 같은 데이터센터급 GPU에서 테스트하여 더 큰 모델(≥7B, 13B)도 굴려보고 싶습니다. “3090아, 너는 잘 싸웠다… 이제 그만 쉬어라.”

---

## 프로젝트 실행 방법

본 레포지토리를 그대로 복제하여 실행하는 것은 불가능합니다.
왜냐하면 본 프로젝트에서 사용한 LLM, 임베딩, 리랭커 모델은 제 개인 PC에서 구동 중이며, API 키를 발급·공유해야만 동일 환경이 재현되기 때문입니다. (보안은 소중하니까요.)

다만, 업로드된 코드와 파이프라인 로직을 참고하시면 동일한 방식으로 RAG 시스템을 재구현하는 것은 충분히 가능합니다. 모델 및 환경 세팅만 본인 상황에 맞게 교체하시면 됩니다.

---

## Acknowledgements

- 본 프로젝트는 **KH정보교육원** 파이널 팀프로젝트 과정 중 진행되었습니다.
- ERP 시스템 CoreFlow를 함께 개발한 팀원들에게 감사드립니다. (제가 맡은 AI 파트는 그 덕에 빛날 수 있었습니다 🙃)
- RAG 및 LLM 오픈소스 생태계(Hugging Face, LangChain, Ollama, Chroma 등)를 만들어 주신 커뮤니티에도 큰 감사를 드립니다.

---

## License

Code © 2025 Jongyun Han (Max). Released under the MIT License. See the LICENSE file for details.

Note: 본 레포지토리에는 **실제 데이터셋은 포함되어 있지 않으며**, 예시 규정 문서는 프로젝트 설명을 위해 생성된 데이터임. 
실제 환경에 적용 시에는 반드시 해당 조직의 내부 규정/데이터를 사용해야 합니다.









