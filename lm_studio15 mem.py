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
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage
from langchain import hub
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory


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
query4="History of Aurangjeb King"
query5="what's 3 plus 5. also whats 3 * 100"

template = '''Answer the following questions as best you can. You have access to the following tools:



Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of tool_names
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


from langchain.memory import ChatMessageHistory
memory = ChatMessageHistory(session_id="test-session")

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

response1=agent_with_chat_history.invoke(
    {"input": "Hi My name is Ashish Choube"},
    config={"configurable": {"session_id": "<foo>"}})
print(response1)
print("#####################")


response2=agent_with_chat_history.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "<foo>"}})
print(response2)

