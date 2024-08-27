# Example: reuse your existing OpenAI setup
from openai import OpenAI
from langchain_openai import ChatOpenAI
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
from langchain.agents import AgentExecutor, create_self_ask_with_search_agent
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_core.prompts import ChatPromptTemplate

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


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y

@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wiki_tool=Tool(name="Wikipedia",
               func=wiki.run,
               description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")


tools=[add,multiply,exponentiate,wiki_tool]

query1 ="what is addition of 5 and 10"
query2="Hi"
query3="what is multiplication of 50 and 210"
query4="Weather in Banglore"
query5="what's 3 plus 5. also whats 3 * 100"

prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant"), 
    ("human", "{input}"), 
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

response=agent_executor.invoke({"input": query1, })
print(response)