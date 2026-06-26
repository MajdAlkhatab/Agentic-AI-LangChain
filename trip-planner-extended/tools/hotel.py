from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

from state import TripState
from mcp_clients import web_search

skiplagged_hotel_executor = None
trivago_executor = None
airbnb_executor = None
hotelzero_executor = None


def init_hotel_executors(all_tools: list):
    global skiplagged_hotel_executor, trivago_executor, airbnb_executor, hotelzero_executor

    skiplagged_hotel_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in all_tools if t.name == "sk_hotels_search"],
        system_prompt="""
        You are a hotel price hunter using Skiplagged.
        Search for the cheapest hotels at the given destination and return:
        hotel name, price per night, and total price for the stay.
        No follow-up questions.
        """,
    )

    trivago_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in all_tools if t.name == "trivago-accommodation-search"],
        system_prompt="""
        You are a hotel price hunter using Trivago.
        Search for the cheapest hotels at the given destination and return:
        hotel name, price per night, and total price for the stay.
        No follow-up questions.
        """,
    )

    airbnb_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in all_tools if t.name == "airbnb_search"],
        system_prompt="""
        You are an accommodation hunter using Airbnb.
        Search for the cheapest listings at the given destination and return:
        listing name, price per night, and total price for the stay.
        No follow-up questions.
        """,
    )

    hotelzero_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in all_tools if t.name in ("find_hotels", "get_price_calendar")],
        system_prompt="""
        You are a hotel price hunter using HotelZero.
        Search for the cheapest hotels at the given destination. Use get_price_calendar
        to find the cheapest nights. Return: hotel name, price per night, best dates.
        No follow-up questions.
        """,
    )


@tool
async def search_hotels(destination: str, nights: str, runtime: ToolRuntime[None, TripState]) -> str:
    """Hotel team-leader: search all hotel sources for the cheapest stay at a given destination.
    Returns the cheapest option found across all sources with price per night and total cost."""
    travelers = runtime.state["travelers"]
    query = f"Find the cheapest hotel in {destination} for {travelers} traveler(s) for {nights} nights."

    results = []
    for executor in [skiplagged_hotel_executor, trivago_executor, airbnb_executor, hotelzero_executor]:
        try:
            response = await executor.ainvoke({"messages": [HumanMessage(content=query)]})
            results.append(response["messages"][-1].content)
        except Exception as e:
            results.append(f"Source failed: {e}")

    combined = "\n\n---\n\n".join(results)
    return f"Hotel results for {destination} ({nights} nights):\n\n{combined}"
