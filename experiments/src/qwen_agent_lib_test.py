from qwen_agent.agents import Assistant
from dotenv import load_dotenv
load_dotenv()
import os, asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from datetime import timedelta

llm_cfg = {
    'model': os.getenv("ollama_model"),

    # Use the endpoint provided by Alibaba Model Studio:
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # Use a custom endpoint compatible with OpenAI API:
    'model_server': os.getenv("ollama_url")+'/v1',  # api_base
    'api_key': 'EMPTY',

    # Other parameters:
    'generate_cfg': {
        # Add: When the response content is `<think>this is the thought</think>this is the answer;
        # Do not add: When the response has been separated by reasoning_content and content.
        'thought_in_content': True,
    },
}

# Define Tools
client = MultiServerMCPClient(
    {
        "RAG_and_weather": {
            "transport": "streamable_http",
            "url": "http://localhost:8000/mcp",
            "timeout": timedelta(seconds=180)
        },
    }
)

mcp_tools = asyncio.run(client.get_tools())

# Define Tools
tools = [
    {'mcpServers': {  # You can specify the MCP configuration file
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
  'code_interpreter',  # Built-in tools
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=mcp_tools)

# Streaming generation
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
print(responses)