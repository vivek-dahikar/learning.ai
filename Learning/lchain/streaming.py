# instead of waiting for entire response to be returned, we can process result as soon as available

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage

model = OllamaLLM(streaming=True, model="llama3.2")

# for chunk in model.stream("what is the capital of India?"):
#     print(chunk, end="", flush=True)

for chunk in model.stream([HumanMessage(content="what is the capital of India?")]):
    print(chunk, end="", flush=True)   