from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACE_API_KEY")
if not api_key:
    st.error("Please set your HUGGINGFACE_API_KEY in the .env file")
    st.stop()
# print(api_key)
model = HuggingFaceEndpoint(
    repo_id="google/flan-t5-base",
    task="text2text-generation",
    huggingfacehub_api_token=api_key
)

st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('LangChain Prompts/template.json')

if st.button('Summarize'):
    try:
        chain = template | model
        result = chain.invoke({
            'paper_input':paper_input,
            'style_input':style_input,
            'length_input':length_input
        })
        st.write(result)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have access to the model and your API key is correct.")