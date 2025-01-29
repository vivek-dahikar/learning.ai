
# Document loader and text splitting

from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
import requests

url = "https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/"

#save it locally
r= requests.get(url)

# extract text from html/
soup = BeautifulSoup(r.text, 'html.parser')
text = soup.get_text()

with open("nvidia.txt", "w") as f:
    f.write(text)

loader= TextLoader("nvidia.txt")
# docs = loader.load()

# print(docs)


# split text into sentences
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex= False
)

final_docs = text_splitter.split_documents(loader.load())

# print(final_docs)

# Summarizing documents using langchain
from langchain_ollama import OllamaLLM
from langchain.chains.summarize import load_summarize_chain 

chat = OllamaLLM(model="llama3.2")
chain = load_summarize_chain(chat, chain_type="map_reduce")
result = chain.invoke({
    "input_documents": final_docs,
})

print(result)






# you can create doc using langchain
# from langchain.schema import Document
# Document(page_content=text, metadata={"url": url})