# Example: reuse your existing OpenAI setup
from openai import OpenAI
from langchain_openai import ChatOpenAI
from ast import mod
from urllib import response
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# Point to the local server

# Point to the local server
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",model="gorilla-llm/gorilla-openfunctions-v2-gguf")

@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a given word."""
    print("Using get_word_length_tool")
    return len(word)

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wiki_tool=Tool(name="Wikipedia",
               func=wiki.run,
               description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")


@tool
def get_upper_word(word: str) -> int:
    """Returns the string in upper form only."""
    print("Using get_upper_word_length")
    return word.upper()

tools=[get_word_length,get_upper_word,wiki_tool]

model_with_tools = llm.bind_tools(tools)

query1 ="what is the length of word Ashish ?"
query2="Hi"
query3="convert Ashish word into upper form"
query3="What happened with Yummo Ice cream in Mumbai"

from langchain_core.messages import HumanMessage
# response = model_with_tools.invoke([HumanMessage(content="Hi!")])
response = model_with_tools.invoke(query3)

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
print("By using Gorila Function and LM Studio")
print("##########################################################")

from langchain_openai import ChatOpenAI

llm1 = ChatOpenAI(
    api_key="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )


@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a given word."""
    print("Using get_word_length_tool")
    return len(word)

tools=[get_word_length,get_upper_word,wiki_tool]

model_with_tools = llm1.bind_tools(tools)

query1 ="what is the length of word Ashish ?"
query2="Hi"

from langchain_core.messages import HumanMessage
# response = model_with_tools.invoke([HumanMessage(content="Hi!")])
response = model_with_tools.invoke(query3)

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
print("WITH Ollama and LLAMA 3")
print("##########################################################")



from langchain_openai import ChatOpenAI

llm2 = ChatOpenAI(
    api_key="ollama",
    model="mistral",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a given word."""
    print("Using get_word_length_tool")
    return len(word)

tools=[get_word_length,get_upper_word,wiki_tool]

model_with_tools = llm2.bind_tools(tools)

query1 ="what is the length of word Ashish ?"
query2="Hi"

from langchain_core.messages import HumanMessage
# response = model_with_tools.invoke([HumanMessage(content="Hi!")])
response = model_with_tools.invoke(query3)

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
print("WITH Ollama and Mistral")
print("##########################################################")

from langchain_openai import ChatOpenAI

llm3 = ChatOpenAI(
    api_key="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a given word."""
    print("Using get_word_length_tool")
    return len(word)

tools=[get_word_length,get_upper_word,wiki_tool]

model_with_tools = llm3.bind_tools(tools)

query1 ="what is the length of word Ashish ?"
query2="Hi"

from langchain_core.messages import HumanMessage
# response = model_with_tools.invoke([HumanMessage(content="Hi!")])
response = model_with_tools.invoke(query3)

print(f"ContentString: {response.content}")
print(f"ToolCalls: {response.tool_calls}")
print("WITH Ollama and Zephyr")
print("##########################################################")

