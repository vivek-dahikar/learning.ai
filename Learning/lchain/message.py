from langchain_ollama import OllamaLLM
chat = OllamaLLM(model="llama3.2")

# answer = chat.invoke("what is the capital of India?")
# print(answer)

# <----------------------------->
# <----------------------------using messages------------------------>

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

text = "what is rag?"
messages  = [
    SystemMessage(content="You are world leading ai engineer"),
    HumanMessage(content=text)
]
result = chat.invoke(messages)
print(result)


# <----------------------------->
# <------------------------using anthropic------------------>

# from langchain_anthropic import ChatAnthropic

# chat = ChatAnthropic(model='claude-3-opus-20240229')
# answer = chat.invoke("what is the capital of India?")
# print(answer)
