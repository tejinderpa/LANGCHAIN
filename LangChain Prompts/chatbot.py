from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

# Load your Hugging Face API key from .env
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_api_key:
    print("Error: Please set your HUGGINGFACE_API_KEY in the .env file")
    exit(1)

# Initialize the model with better parameters
model = HuggingFaceEndpoint(
    repo_id="microsoft/DialoGPT-medium",
    task="text-generation",
    temperature=0.7,
    max_new_tokens=100,
    huggingfacehub_api_token=hf_api_key
)

def format_chat_history(messages):
    """Format chat history into a prompt string."""
    # For DialoGPT, we only need the last user message
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break
    
    if last_user_message is None:
        return ""
    return last_user_message

def main():
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")
    print("Type your message and press Enter to chat.")
    print("-" * 50)

    chat_history = [
        SystemMessage(content="You are a helpful, friendly, and knowledgeable AI assistant. Provide clear and concise responses.")
    ]

    while True:
        try:
            user_input = input('\nYou: ').strip()
            
            if user_input.lower() == 'exit':
                print("\nGoodbye! Thanks for chatting!")
                break
                
            if not user_input:
                print("Please enter a message.")
                continue

            # Add user message to history
            chat_history.append(HumanMessage(content=user_input))
            
            # Format the prompt
            prompt = format_chat_history(chat_history)
            
            # Get AI response
            print("\nAI is thinking...")
            result = model.invoke(prompt)
            
            # Add AI response to history
            chat_history.append(AIMessage(content=result))
            print(f"\nAI: {result}")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
