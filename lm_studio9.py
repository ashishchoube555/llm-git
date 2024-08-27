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


llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",model="gorilla-llm/gorilla-openfunctions-v2-gguf")

@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a given word."""
    print("Using get_word_length_tool")
    return len(word)

@tool
def get_upper_word(word: str) -> int:
    """Returns the string in upper form only."""
    print("Using get_upper_word_length")
    return word.upper()


wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wiki_tool=Tool(name="Wikipedia",
               func=wiki.run,
               description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")


tools=[get_word_length,get_upper_word,wiki_tool]

# Now we are using create react agent so no need to use bind tools

query1 ="what is the length of word Ashish ?"
query2="Hi"
query3="convert Ashish word into upper form"
query4="What happened with Yummo Ice cream in Mumbai"

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(llm, tools)

response = agent_executor.invoke({"messages": [HumanMessage(content=query4)]})
print(response["messages"])

print("By using Gorila Function and LM Studio")
print("##########################################################")

llm1 = ChatOpenAI(
    api_key="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

agent_executor1 = create_react_agent(llm1, tools)

response1 = agent_executor1.invoke({"messages": [HumanMessage(content=query4)]})
print(response1["messages"])

print("WITH Ollama and LLAMA 3")
print("##########################################################")


llm2 = ChatOpenAI(
    api_key="ollama",
    model="mistral",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

agent_executor2 = create_react_agent(llm2, tools)

response2 = agent_executor2.invoke({"messages": [HumanMessage(content=query4)]})
print(response2["messages"])

print("WITH Ollama and Mistral")
print("##########################################################")

llm3 = ChatOpenAI(
    api_key="ollama",
    model="llama3",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

agent_executor3 = create_react_agent(llm3, tools)

response3 = agent_executor3.invoke({"messages": [HumanMessage(content=query4)]})
print(response3["messages"])
print("WITH Ollama and Zephyr")
print("##########################################################")

