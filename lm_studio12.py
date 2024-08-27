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
from langchain.agents import AgentExecutor, create_react_agent
import os
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent

# llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",model="gorilla-llm/gorilla-openfunctions-v2-gguf")

# llm = ChatOpenAI(
#     api_key="ollama",
#     model="llama3",
#     base_url="http://localhost:11434/v1",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     )

llm = ChatOpenAI(
    api_key="ollama",
    model="mistral",
    base_url="http://localhost:11434/v1",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )


# llm = ChatOpenAI(
#     api_key="ollama",
#     model="zephyr",
#     base_url="http://localhost:11434/v1",
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
#     )

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


os.environ["TAVILY_API_KEY"] = "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"

# Set up tools there
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search,max_results=1)
# search_results = tavily_tool.invoke("what happend with India England T20 Match?")
# print(search_results)


tools=[get_word_length,get_upper_word,tavily_tool,wiki_tool]

# Now we are using create react agent so no need to use bind tools

query1 ="what is the length of word Ashish"
query2="Hi"
query3="convert Ashish word into upper form"
query4="Weather in Banglore"
query5="Is india won in world cup in 2024?"

from langchain_core.messages import HumanMessage
from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")

agent = create_tool_calling_agent(llm, tools, prompt)

# agent = create_react_agent(llm, tools,prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True,Callbacks=CallbackManager)
response = agent_executor.invoke({"input": [HumanMessage(content=query1)]})
print(response)

print("By using Ollama and Mistral")
print("##########################################################")
