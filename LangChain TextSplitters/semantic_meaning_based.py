from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

text = """
Sure! Here are **6 lines**, each with a **different semantic meaning** (purpose or interpretation):

1. **Factual Statement**:
   *The Earth orbits the Sun once every 365.25 days.*

2. **Command**:
   *Please reboot the server before running the update.*

3. **Question**:
   *Why is the neural network overfitting on the training data?*

4. **Exclamation/Emotion**:
   *Wow! That generative model just wrote a whole novel!*

5. **Hypothetical/Conditional**:
   *If we fine-tune the model, it might improve accuracy.*

6. **Metaphor**:
   *The algorithm danced gracefully through the data like a seasoned performer.*

Let me know if you'd like these tailored to a specific theme like AI, love, war, or education!


"""

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chunker = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type = "standard-deviation",
    breakpoint_threshold_amount =  1
)

chunks = chunker.create_documents([text])
print(chunks[0])

