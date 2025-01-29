

# getting structured outputs from a model.

from typing import List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field



model = OllamaLLM(model="llama3.2")

class Joke(BaseModel):
    setup: str = Field(description="Joke setup")
    punchline: str = Field(description="Joke punchline")

# <--------------- DIRECT WAY TO DO IT (Note: ollama model doesn't support it)----------------------->

# structure_llm = model.with_structured_output(Joke)
# result = structure_llm.invoke("Tell me a joke about ai")
# print(result.setup)
# print(result.punchline)


# <------------------lONG WAY TO DO IT------------------------->


class Jokes(BaseModel):
    jokes: List[Joke] = Field(description="List of jokes")


parser = PydanticOutputParser(pydantic_object=Jokes)

template = "Answer the user query, \n{formate_instruction} \n{query}"

System_Message_Prompt = SystemMessagePromptTemplate.from_template(template)
Chat_Prompt = ChatPromptTemplate.from_messages([System_Message_Prompt])

message = Chat_Prompt.invoke({
    "query": "Tell me a joke about ai",
    "formate_instruction": parser.get_format_instructions()
})

result = model.invoke(message)
print(result)

try:
    jokes = parser.parse(result)
    for joke in jokes.jokes:
        print(joke.setup)
        print(joke.punchline)
except Exception as e:
    print(e)
    print(result)