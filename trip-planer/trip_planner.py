#  _____                 _    _____ _                     
# |_   _|___ ___ _ _ ___| |  |  _  | |___ ___ ___ ___ ___ 
#   | | |  _| .'| | | -_| |  |   __| | .'|   |   | -_|  _|
#   |_| |_| |__,|\_/|___|_|  |__|  |_|__,|_|_|_|_|___|_|  
# 
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage, ToolMessage
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentState, create_agent
from mcp.types import CallToolResult, TextContent
from langchain.tools import tool, ToolRuntime
from mcp.shared.exceptions import McpError
from langgraph.types import Command
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, Any
import asyncio
import sys

load_dotenv()

if sys.platform == "win32":
    if not isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    if "ipykernel" in sys.modules:
        sys.stderr = sys.__stderr__

RETRYABLE_MCP_CODES = {-32603}

class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt+1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt+1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )

async def init_mcp_tools():
    client = MultiServerMCPClient(
        {
            "travel_server": {
                    "transport": "streamable_http",
                    "url": "https://mcp.kiwi.com"
                }
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )
    return await client.get_tools()

try:
    loop = asyncio.get_running_loop()
    tools = loop.run_until_complete(init_mcp_tools())
except RuntimeError:
    tools = asyncio.run(init_mcp_tools())

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}

@tool
def get_weather(query: str) -> Dict[str, Any]:
    """Search for weather forecast and best time of year to visit a destination"""
    try:
        return tavily_client.search(f"weather forecast best time to visit {query}")
    except Exception as e:
        return {"error": str(e)}

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def query_chinook_db(query: str) -> str:
    """Query the Chinook database for playlist information"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"

class TripState(AgentState):
    origin: str
    destination: str
    budget: str
    interests: str

current_date = datetime.now().strftime("%d/%m/%Y")

# Travel agent
travel_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt=f"""
    You are a travel agent. Search for flights to the desired destination.
    The current date is {current_date}. You MUST ensure any departure dates you search for are strictly in the future.
    You are not allowed to ask any more follow up questions, you must find the best flight options.
    Only look for one ticket, one way.
    If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again with a different future date.
    Once you have found the best options, let the user know your shortlist of options.
    """
)

# Hotel agent
hotel_agent = create_agent(
    model="gpt-5-nano",
    tools=tools,
    system_prompt=f"""
    You are a hotel and accommodation specialist. Search for places to stay at the desired destination.
    The current date is {current_date}. Ensure any check-in dates you search for are in the future.
    Find the best accommodation options that fit within a reasonable budget.
    If the MCP tool fails, returns malformed output, or does not give usable results, try the tool again.
    Summarize your top hotel picks for the user.
    """
)

# Activities agent
activities_agent = create_agent(
    model="gpt-5-nano",
    tools=[web_search],
    system_prompt="""
    You are an activities specialist. Search for things to do at the destination that match the traveller's interests.
    You are not allowed to ask any more follow up questions, you must find the best activity options.
    You may need to make multiple searches to iteratively find the best options.
    You have a suggested limit of 12 web searches. Count every web_search call you make.
    """
)

# Weather agent
weather_agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    system_prompt="""
    You are a weather specialist. Search for the weather forecast and best time of year to visit the destination.
    Once you have the information, give the user a short summary of conditions and what to pack.
    """
)

# Culture agent
culture_agent = create_agent(
    model="gpt-5-nano",
    tools=[], 
    system_prompt="""
    You are a culture researcher. Relying entirely on your own internal knowledge, find what travellers from the same origin tend to enjoy at the destination.
    Use this to give a quirky cultural recommendation for the traveller.
    """
)

# Playlist agent
playlist_agent = create_agent(
    model="gpt-5-nano",
    tools=[query_chinook_db],
    system_prompt="""
    You are a playlist specialist. Query the sql database and curate the perfect playlist for an airplane ride.
    Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
    If you run into errors when querying the database, try to fix them by making changes to the query.
    Do not come back empty handed, keep trying to query the db until you find a list of songs.

    This is a SQLite database. Before writing any data queries, first discover the schema.
    """
)
# tools for the main coordinator 
@tool
async def search_flights(runtime: ToolRuntime[None, TripState]) -> str:
    """Travel agent searches for flights to the desired destination."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    response = await travel_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
    return response['messages'][-1].content

@tool
async def search_hotels(runtime: ToolRuntime[None, TripState]) -> str:
    """Hotel agent searches for accommodations at the desired destination."""
    destination = runtime.state["destination"]
    response = await hotel_agent.ainvoke({"messages": [HumanMessage(content=f"Find hotels in {destination}")]})
    return response['messages'][-1].content

@tool
def search_activities(runtime: ToolRuntime[None, TripState]) -> str:
    """Activities agent finds things to do at the destination matching the traveller's interests."""
    destination = runtime.state["destination"]
    interests = runtime.state["interests"]
    query = f"Find {interests} activities in {destination}"
    response = activities_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool
async def check_weather(runtime: ToolRuntime[None, TripState]) -> str:
    """Weather agent checks the forecast and best time to visit the destination."""
    destination = runtime.state["destination"]
    response = await weather_agent.ainvoke({"messages": [HumanMessage(content=f"Weather and best time to visit {destination}")]})
    return response['messages'][-1].content

@tool
def check_culture(runtime: ToolRuntime[None, TripState]) -> str:
    """Culture agent finds what travellers from the same origin tend to enjoy."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    response = culture_agent.invoke({"messages": [HumanMessage(content=f"Find cultural insights for travellers from {origin} traveling to {destination}")]})
    return response['messages'][-1].content

@tool
def suggest_playlist(runtime: ToolRuntime[None, TripState]) -> str:
    """Playlist agent curates the perfect playlist for the airplane ride."""
    query = "Find relaxing tracks for an airplane ride playlist"
    response = playlist_agent.invoke({"messages": [HumanMessage(content=query)]})
    return response['messages'][-1].content

@tool
def update_state(origin: str, destination: str, budget: str, interests: str, runtime: ToolRuntime[None, TripState]) -> str:
    """Update the state when you know all of the values: origin, destination, budget, interests.
    This tool must be called alone, without any other tool calls. It must complete and return to make,
    the information available to other tools."""
    return Command(update={
        "origin": origin,
        "destination": destination,
        "budget": budget,
        "interests": interests,
        "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)]}
        )


#  MAIN COORDINATOR 
coordinator = create_agent(
    model="gpt-5-nano",
    tools=[search_flights, search_hotels, search_activities, check_weather, check_culture, suggest_playlist, update_state],
    state_schema=TripState,
    system_prompt="""
    You are a trip planner.
    First find all the information you need to update the state. When you have the information, update the state.
    Once that has completed and returned, you can delegate the tasks
    to your specialists for flights, hotels, activities, weather, culture, and playlists.
    Once you have received their answers, coordinate the perfect trip for the traveller.
    No follow-up question is needed.
    """
)

async def main():
    response = await coordinator.ainvoke(
        {
            "messages": [HumanMessage(content="I'm from Copenhagen and I'd like to visit Tokyo, budget $2000, I'm into food and hiking")],
        },
        config={"tags": ["TP"], "recursion_limit": 40},
    )
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())