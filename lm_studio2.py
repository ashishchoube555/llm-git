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
llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio",model="gorilla-llm/gorilla-openfunctions-v2-gguf")

@tool
def get_word_length(word: str) -> int:
    """Returns the exact length of a given word."""
    print("Using get_word_length_tool")
    return len(word)

tools=[get_word_length]

query = "what is the length of word Ashish ?"

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant you have to use above function get_word_length and find out the length of given word with help of above functions"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

response=agent_executor.invoke({"input": query})
print(response)

