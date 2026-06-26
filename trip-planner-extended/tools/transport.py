from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

from state import TripState
from mcp_clients import web_search

_prompt = """
You are a transport price estimator. Search the web for the estimated cost
of the specified transport type from the airport to the city center.
Return: transport type, estimated price, and any useful notes.
No follow-up questions.
"""

taxi_executor = create_agent(model="gpt-5-nano", tools=[web_search], system_prompt=_prompt)
bus_executor = create_agent(model="gpt-5-nano", tools=[web_search], system_prompt=_prompt)
uber_executor = create_agent(model="gpt-5-nano", tools=[web_search], system_prompt=_prompt)
local_app_executor = create_agent(model="gpt-5-nano", tools=[web_search], system_prompt=_prompt)


@tool
async def search_transport(runtime: ToolRuntime[None, TripState]) -> str:
    """Transport team-leader: estimate airport-to-city-center cost for all transport types."""
    destination = runtime.state["destination"]

    queries = [
        (taxi_executor,     f"taxi price from airport to city center in {destination}"),
        (bus_executor,      f"public bus from airport to city center in {destination} price"),
        (uber_executor,     f"Uber price from airport to city center in {destination}"),
        (local_app_executor,f"local ride app airport transfer price in {destination}"),
    ]

    results = []
    for executor, query in queries:
        try:
            response = await executor.ainvoke({"messages": [HumanMessage(content=query)]})
            results.append(response["messages"][-1].content)
        except Exception as e:
            results.append(f"Source failed: {e}")

    return "\n\n---\n\n".join(results)
