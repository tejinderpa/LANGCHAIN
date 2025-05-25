import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_API_TOKEN")

res = requests.post(
    "https://api-inference.huggingface.co/models/google/flan-t5-small",
    headers={"Authorization": f"Bearer {token}"},
    json={"inputs": "Translate English to French: I love programming."}
)

print(res.status_code)
print(res.json())
