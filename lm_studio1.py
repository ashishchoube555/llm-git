# Example: reuse your existing OpenAI setup
from openai import OpenAI
from langchain_openai import ChatOpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="gorilla-llm/gorilla-openfunctions-v2-gguf",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ],
  temperature=0.7,
)

# print(completion.choices[0].message)


from langchain_core.tools import tool
@tool
def add(first_int: int, second_int: int) -> int:
    "Add two integers."
    return first_int + second_int

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def exponentiate(base: int, exponent: int) -> int:
    "Exponentiate the base to the exponent power."
    return base**exponent

from operator import itemgetter
from typing import Dict, List, Union

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)

tools = [multiply, exponentiate, add]

llm_with_tools = completion.bind_tools(tools)
# print(llm_with_tools)

# print("#################")
# print("#################")
# print("#################")


tool_map = {tool.name: tool for tool in tools}
# print(tool_map)
# print("#################")
# print("#################")
# print("#################")


def call_tools(msg: AIMessage) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools

# print(chain)
# querry1="what is addition of 10 and 20"
# querry2="what is multiplicatopn of 500 and 1000"
# querry3="add a million plus a billion and cube thirty-seven"
querry4="What is addition of 5 and 10 "
query5 = "What is 3 * 12? Also, what is 11 + 49?"

response1=chain.invoke(
    query5
)
print("Response1")
print(response1)

print("############################################################")


query = "What is 3 * 12? Also, what is 11 + 49?"

response2=llm_with_tools.invoke(query).tool_calls
print(response2)

print("############################################################")