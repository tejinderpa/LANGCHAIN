from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
# print(os.getenv("HUGGINGFACE_API_TOKEN"))

# Initialize the model with correct configuration
model = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/flan-t5-small",
    task="text2text-generation",
    temperature=0.7,
    max_new_tokens=150,
    do_sample=True,
    top_p=0.95,
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

parser = StrOutputParser()

# Create a more specific prompt template

# Create the chain
chain = prompt | model | parser

# Load and process the text file
docs = TextLoader(file_path="LangChain DocumentLoaders/text.txt", encoding="utf-8").load()
print(chain.invoke({"docs": docs[0].page_content}))

"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language="python", 
    chunk_size=350, 
    chunk_overlap=0
)

splits = splitter.split_text(text)

for split in splits:
    print(split)
    print("-"*100)

