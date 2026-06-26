#  _____                 _    _____ _                     
# |_   _|___ ___ _ _ ___| |  |  _  | |___ ___ ___ ___ ___ 
#   | | |  _| .'| | | -_| |  |   __| | .'|   |   | -_|  _|
#   |_| |_| |__,|\_/|___|_|  |__|  |_|__,|_|_|_|_|___|_|  
# 

import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from state import TripState
from mcp_clients import init_mcp_tools

from tools.flight import init_flight_executor, search_flights
from tools.hotel import init_hotel_executors, search_hotels
from tools.transport import search_transport
from tools.activity import init_activity_executors, search_activities
from tools.currency import init_currency_executors, search_currency

load_dotenv()


@tool
def update_state(origin: str, travelers: str, runtime: ToolRuntime[None, TripState]) -> Command:
    """Update state with origin and number of travelers from the user's input.
    Call this first, before any other tool."""
    return Command(update={
        "origin": origin,
        "travelers": travelers,
        "messages": [ToolMessage("State updated: origin and travelers set.", tool_call_id=runtime.tool_call_id)],
    })


@tool
def update_phase1_result(
    destination: str,
    trip_duration_nights: str,
    chosen_flight: str,
    chosen_hotel: str,
    total_cost: str,
    runtime: ToolRuntime[None, TripState],
) -> Command:
    """Save the cheapest flight+hotel combination after Phase 1 comparison.
    Call this alone, after comparing all flight and hotel results.
    total_cost = flight price + (hotel price per night × nights)."""
    return Command(update={
        "destination": destination,
        "trip_duration_nights": trip_duration_nights,
        "chosen_flight": chosen_flight,
        "chosen_hotel": chosen_hotel,
        "total_cost": total_cost,
        "messages": [ToolMessage("Phase 1 complete. Cheapest plan saved.", tool_call_id=runtime.tool_call_id)],
    })


MANAGER_PROMPT = """
You are a cheap travel deal hunter. Your goal is to find the absolute cheapest
complete trip for the user.

Follow this exact sequence:

PHASE 1 — Find the cheapest flight + hotel combination:
1. Call update_state with origin and travelers from the user's message.
2. Call search_flights to get a list of cheap destinations and flight prices.
3. For each promising destination, call search_hotels(destination, nights="3") to get hotel prices.
4. Compute total_cost = flight_price + (hotel_price_per_night × nights) for each combination.
5. Call update_phase1_result with the single cheapest combination. Call this tool alone.

PHASE 2 — Enrich the chosen destination:
6. Call search_transport, search_activities, and search_currency in parallel.

FINAL OUTPUT:
Produce a complete, clear trip plan showing:
- Destination and why it is the cheapest
- Flight details and price
- Hotel details and price per night
- Total cost (flight + hotel)
- Transport options from airport to city center
- Top activities and weather
- Currency exchange rate

No follow-up questions. Present the best deal directly.
"""


async def main():
    all_tools = await init_mcp_tools()

    # Initialise all executors that depend on MCP tools
    init_flight_executor(all_tools)
    init_hotel_executors(all_tools)
    init_activity_executors([t for t in all_tools if t.name.startswith("get_")])
    init_currency_executors([t for t in all_tools if t.name in ("get_rates", "convert", "list_currencies")])

    manager = create_agent(
        model="gpt-5-nano",
        tools=[
            update_state,
            update_phase1_result,
            search_flights,
            search_hotels,
            search_transport,
            search_activities,
            search_currency,
        ],
        state_schema=TripState,
        system_prompt=MANAGER_PROMPT,
    )

    response = await manager.ainvoke(
        {"messages": [HumanMessage(content="From Copenhagen, 2 people")]},
        config={"recursion_limit": 50},
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
