from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# text = """Hello, my name is John Doe. I am a software engineer. I live in San Francisco. I like to code in Python and Java.I have a dog and a cat.
# I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.
# I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.
# I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.
# I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.
# I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.I have a dog and a cat.
# """
loader = PyPDFLoader("LangChain TextSplitters/finalpdf.pdf")
docs = loader.load()
splitter = CharacterTextSplitter(separator=" ", chunk_size=100, chunk_overlap=0)

doc = splitter.split_documents(docs)
print(len(doc))

for i in doc[0:10]:
    print(i.page_content)
    print("--------------------------------")
