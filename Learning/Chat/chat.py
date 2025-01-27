from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OllamaLLM(model="llama3.2")

template = """
Answer the following question based on the document content:
Document Content: {document_content}
Question: {question}
Answer:
"""

prompt = PromptTemplate(input_variables=["document_content", "question"], template=template)

chain = LLMChain(llm=llm, prompt=prompt)

document_content = "I am vivek. I am a data scientist."
question = "Who M I?"

answer = chain.run(document_content=document_content, question=question)

print("Answer:", answer)
