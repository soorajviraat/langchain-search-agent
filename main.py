from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [
    TavilySearch(
        name="Tavily Search",
        description="Useful for when you need to find information about current events or the current state of the world. "
        "Input should be a search query.",
    )
]
llm = ChatOpenAI(model="gpt-4",temperature=0)
react_prompt = hub.pull("hwchase17/react")
react_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
result = agent_executor.invoke(
    {"input": "search for 3 job postings for AI developer using langchain in the bay area on linkedin and list their details"}
)
print(result)



def main():
    print("Hello from search-agent!")


if __name__ == "__main__":
    main()
