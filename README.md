# CoreFlow AI — RAG / Agent / MCP

The AI component of CoreFlow, an ERP system built as the final project at KH Information Educational Institute.

https://github.com/YunSangsoo/COREFLOW_FRONTEND

https://github.com/YunSangsoo/COREFLOW_BACKEND

## Repository Guidelines

Contributors should review [AGENTS.md](AGENTS.md)
 for structure, tooling, and workflow expectations before pushing changes.

## Environment Setup

Copy `.env.example` to `.env`, then fill in local URLs and API keys for the embedding, rerank, geocode, and Ollama services before running the agent stack.

---

## Project Overview

This project implements an in-house AI chatbot for CoreFlow (the ERP final project at KH Information Educational Institute), which is the part I owned.

The goal was to enable the chatbot to answer questions about **internal company policies** using RAG (Retrieval-Augmented Generation).

RAG augments a user query by retrieving and attaching relevant documents so that an LLM can reference them when generating answers. Its advantages include:
- Incorporating up-to-date information into model responses
- Reducing hallucinations
- Answering questions in specific domains (e.g., internal policies)
- No fine-tuning required
- High efficiency & scalability: instead of fine-tuning a large model to “contain” all knowledge, you can use a smaller model
- Lower GPU cost compared to running very large models

The base LLM was OpenAI’s small open-source model [gpt-oss-20b](https://openai.com/index/introducing-gpt-oss/) served via [Ollama](https://ollama.com/). For embeddings we used [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B#evaluation), and for reranking [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B#evaluation). All models ran locally on the following machine:
- **CPU**: AMD Ryzen 5800X3D
- **GPU**: RTX 3090
- **RAM**: 128 GB

![chat screenshot](<images/coreflowaichatexample.png>)
*Figure 1. CoreFlow AI chat screenshot*

---

## Tech Stack

- **Language**: Python
- **LLM / Embedding / Reranker**: gpt-oss-20b / Qwen3-Embedding-4B / Qwen3-Reranker-4B
- **LLM stack**: Ollama, LangChain, Transformers, Sentence Transformers, Chroma
- **Data processing**: Pydantic, pandas
- **Web framework**: FastAPI
- **Dev environment**: Windows 10/11, Ubuntu Desktop 24.04 LTS, VS Code, Jupyter Notebook

---

## Problem

While building the ERP system, we needed an internal chatbot that could answer questions about company policies.

This is not feasible with a plain LLM API alone, underscoring the need for **RAG**.

---

## Dataset

We generated approximately 20 A4 pages of policy documents using Google Gemini’s Deep Research feature. The set consists of Chapters 1 through 11.

---

## Methodology & Approach

Instead of the common RAG pipeline—uniform chunking into a vector DB, direct retrieval with the raw user query, and passing only those hits—we applied a new idea:

1. **Splitting**
    - **Class 1 — Sentence-level units for similarity search**: First, we split all documents by **sentence** and stored them in the vector DB.
    - **Class 2 — Parent documents**: We also created **parent documents** (chapter/section/paragraph level) that contain those sentences and saved all of them to CSV.
2. **Retrieval**
    - **Convert to a vector-search-optimized query**: The user’s question is rewritten by the LLM into a query optimized for vector search. e.g., “Tell me about CoreFlow’s vacation policy.” → “CoreFlow vacation policy”
    - **Vector DB search**: Using the optimized query, we retrieve similar sentences with MMR (k: 10, fetch_k: 50).
    - **Parent document lookup**: For the retrieved sentences, we fetch their parent documents and deduplicate.
    - **Reranker**: The reranker scores the original user question against the candidate parent documents.
3. **Attachment**
    - We attach the top 5 parent documents (by reranker score) to the user query and pass them to the model.

As shown in Figure 1, this approach successfully enabled the model to answer questions about internal policies.

Additionally, we converted the RAG pipeline into an MCP tool and made the base LLM into an **Agent** so it can invoke tools when needed.

---

## Results & Key Observations

- **Improved answer quality vs. naive chunking**: The sentence-level + parent-document + reranker pipeline produced more precise, contextual answers than plain chunk-based retrieval.
- **Reduced hallucination**: Fewer unfounded responses; a higher proportion of answers grounded in the actual policy text.
- **Better retrieval accuracy**: The MMR + reranker combo surfaced documents that better reflected user intent.
- **Practicality**: By wrapping the base LLM as an Agent, the chatbot could handle both policy Q&A and small talk.
- **Local feasibility validated**: On a single RTX 3090, combining gpt-oss-20b with the Qwen models proved sufficient to stand up a practical RAG pipeline.

---

## Conclusion & Future Work

We successfully implemented an internal-policy chatbot for the ERP system using RAG. Compared to simple chunk-splitting, our strategy (sentence-level splitting + parent-document mapping + reranking) yielded more accurate and contextual answers. We also validated that a local setup (RTX 3090) can run gpt-oss-20b with Qwen models to build a working RAG pipeline.

Future directions:
- **Scale to real-world corpora**: Validate on large, real corporate policy/manual/knowledge bases rather than simulated policies.
- **Multimodal expansion**: Extend RAG beyond text to handle images/tables and other formats.
- **Model optimization & slimming**: Apply quantization, distillation, and caching to speed up local inference.
- **UX improvements**: Better chatbot UI; intent detection (e.g., auto-switching among summary/FAQ/search modes).
- **Security & access control**: Strengthen authorization and logging for internal data.
- **High-end validation**: We ran the pipeline on an RTX 3090 and heard its VRAM cry. Next, we’d like to test on data-center GPUs (A100/H100) and try larger models (≥7B, 13B). “3090, you fought bravely—now rest.”

---

## How to Run

You cannot run this repository as-is.

The LLM, embedding, and reranker services are hosted on my personal PC, and reproducing the exact environment would require issuing and sharing API keys (security matters!).

However, the uploaded code and pipeline logic are sufficient to re-implement the same RAG approach. Replace the models and environment with ones that suit your setup.

---

## Acknowledgements

- This work was carried out as part of the **KH Information Educational Institute** final team project.
- Thanks to my CoreFlow teammates—your work made this AI part shine 🙃.
- Deep appreciation to the open-source RAG/LLM community (Hugging Face, LangChain, Ollama, Chroma, etc.).

---

## License

Code © 2025 Jongyun Han (Max). Released under the MIT License. See the LICENSE file for details.

Note: This repository **does not include any real datasets.** The sample policy documents were generated to illustrate the project. For real deployments, always use your organization’s internal policies/data.