from dotenv import load_dotenv
load_dotenv()
import asyncio
from src.agents.mcp import server

async def main():
    compressed = await server.run_query_convert_agent("CoreFlow 휴가 규정")
    print("compressed", compressed)
    embeddings = server.CustomEmbeddings()
    vector_store = server.Chroma(
        persist_directory=server.DB_PATH,
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 50})
    docs_list = retriever.invoke(compressed)
    parent_docs = server.search_parent_docs(docs_list, server.parent_docs_list)
    print("docs count", len(parent_docs))
    print("first doc length", len(parent_docs[0]))
    print(parent_docs[0][:300])
    print("max length", max(len(doc) for doc in parent_docs))

asyncio.run(main())
