from langchain_ollama import OllamaLLM
from langchain_core.prompts.chat import ChatPromptTemplate


chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are world leading ai engineer"),
    ("human", "{text}")
])

result = chat_prompt_template.invoke({
    "text": "what is rag?"
})
# print(type(result))

model = OllamaLLM(model="llama3.2")
ai_llm_result = model.invoke(result)
print(ai_llm_result)
