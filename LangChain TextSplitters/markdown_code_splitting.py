from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """😄 That's a classic tech pun!

**Joke:**
**Q:** Why did the AI break up with its computer?
**A:** Because it couldn't handle the constant **CTRL + ALT + DELETE!**

🧠 **Explanation:**
"Ctrl + Alt + Delete" is a key combination often used to:

* Restart a computer
* Open Task Manager (on Windows)
* Force quit processes

So the joke is that the AI found the constant interruptions or restarts too stressful—just like relationship drama. 💔💻

Want me to generate more AI or programming jokes like this?


"""
splitter = RecursiveCharacterTextSplitter.from_language(
    language="markdown", 
    chunk_size=350, 
    chunk_overlap=0
)

splits = splitter.split_text(text)

for split in splits:
    print(split)
    print("-"*100)

