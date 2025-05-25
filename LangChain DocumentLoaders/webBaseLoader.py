from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Verify token is loaded
token = os.getenv("HUGGINGFACE_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

model = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-small",
    task="text2text-generation",
    temperature=0.7,
    max_new_tokens=150,
    huggingfacehub_api_token=token
)

prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question':'What is the product that we are talking about?', 'text':docs[0].page_content}))