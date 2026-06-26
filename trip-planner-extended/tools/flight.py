from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

from state import TripState

# Executor is created after MCP tools are loaded (see main.py)
skiplagged_flight_executor = None


def init_flight_executor(sk_tools: list):
    global skiplagged_flight_executor
    skiplagged_flight_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in sk_tools if t.name == "sk_destinations_anywhere"],
        system_prompt="""
        You are a flight deal hunter. Given an origin city, use sk_destinations_anywhere
        to find the cheapest destinations to fly to. Return a clear list of:
        destination, price (EUR), and departure date.
        Focus only on the cheapest options. No follow-up questions.
        """,
    )


@tool
async def search_flights(runtime: ToolRuntime[None, TripState]) -> str:
    """Flight team-leader: find cheapest destinations and prices from the origin city."""
    origin = runtime.state["origin"]
    travelers = runtime.state["travelers"]
    response = await skiplagged_flight_executor.ainvoke({
        "messages": [HumanMessage(
            content=f"Find the cheapest destinations to fly to from {origin} for {travelers} traveler(s)."
        )]
    })
    return response["messages"][-1].content
