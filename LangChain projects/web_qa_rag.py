from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set USER_AGENT for Hugging Face
os.environ["USER_AGENT"] = "LangChain Web QA App"

# Initialize the HuggingFace model
model = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    temperature=0.5,
    model_kwargs={"max_length": 512},
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
)

# Create embeddings with explicit model name
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create the prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

def load_and_process_webpage(url):
    # Load the webpage
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore

def answer_question(url, question):
    # Load and process the webpage
    vectorstore = load_and_process_webpage(url)
    
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create the chain
    chain = prompt | model | StrOutputParser()
    
    # Get the answer
    answer = chain.invoke({"context": context, "question": question})
    
    return answer

if __name__ == "__main__":
    # Example usage
    url = input("Enter the webpage URL: ")
    question = input("Enter your question: ")
    
    try:
        answer = answer_question(url, question)
        print("\nAnswer:", answer)
    except Exception as e:
        print(f"An error occurred: {str(e)}") 