import os, requests
from dotenv import load_dotenv
load_dotenv()
headers = {"authorization": os.getenv("myserver_api_key")}
url = os.getenv("myserver_url") + ":8000/embed"
for sample in ("", " ", "test"):
    resp = requests.post(url, json={"texts": [sample], "is_query": True}, headers=headers)
    print(repr(sample), resp.status_code)
    print(resp.text)
