from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """My name is Nitish
I am 35 years old
I live in gurgaon
How are you
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)

doc = splitter.split_text(text)

for i in doc:
    print(i)
    print("--------------------------------")

# when chunk size was 10
# output was :
# My name is 
# #---------------------
# Nitish
# #---------------------    
# I am 35 
# #---------------------
# years old
# #---------------------
# I live in 
# #---------------------
# gurgaon
# #---------------------
# How are 
# #---------------------
# you
# #---------------------

# when chunk size was 25
# output was :
# My name is Nitish
#---------------------
# I am 35 years old
#---------------------
# I live in gurgaon
#---------------------
# How are you
#---------------------


# when chunk size was 50
# output was :
# My name is Nitish
# I am 35 years old
# --------------------
# I live in gurgaon
# How are you



