# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from dotenv import load_dotenv
# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )


# res = llm.invoke("Who is Tejinderpal singh")
# print(res.content)

from langchain_huggingface import ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

chat_model = ChatHuggingFace(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

response = chat_model.invoke("Who is Tejinderpal singh?")
print(response.content)
