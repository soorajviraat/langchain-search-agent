from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from prompts import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [
    TavilySearch(
        name="Tavily Search",
        description="Useful for when you need to find information about current events or the current state of the world. "
        "Input should be a search query.",
    )
]
llm = ChatOpenAI(model="gpt-4",temperature=0)
# react_prompt = hub.pull("hwchase17/react")
outputParser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_instructions = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad", "format_instructions"],
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
).partial(format_instructions=outputParser.get_format_instructions())


react_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_instructions)
agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)
parse_output = RunnableLambda(
    lambda x: outputParser.parse(x["output"]),
)
chain = agent_executor | parse_output
result = chain.invoke(
    {"input": "search for 3 job postings for AI developer using langchain in the bay area on linkedin and list their details"}
)
print(result)



def main():
    print("Hello from search-agent!")


if __name__ == "__main__":
    main()
