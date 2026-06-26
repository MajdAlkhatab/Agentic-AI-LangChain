from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

from state import TripState
from mcp_clients import web_search

openmeteo_executor = None
tavily_weather_executor = None
activities_executor = None


def init_activity_executors(weather_tools: list):
    global openmeteo_executor, tavily_weather_executor, activities_executor

    openmeteo_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in weather_tools if t.name in ("get_weather_byDateTimeRange", "get_current_weather")],
        system_prompt="""
        You are a weather specialist. Get the weather forecast for the destination.
        Return a short summary: expected temperature, conditions, and what to pack.
        No follow-up questions.
        """,
    )

    tavily_weather_executor = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt="""
        You are a weather specialist. Search the web for the weather forecast at the destination.
        Return a short summary: expected temperature, conditions, and what to pack.
        No follow-up questions.
        """,
    )

    activities_executor = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt="""
        You are an activities specialist. Search the web for cheap or free things to do
        at the destination. Return a short list of top activities with estimated costs.
        No follow-up questions.
        """,
    )


# Culture executor uses model knowledge only — no tools needed
culture_executor = create_agent(
    model="gpt-5-nano",
    tools=[],
    system_prompt="""
    You are a culture researcher. Using your own knowledge, give a short, useful
    cultural tip for a traveller visiting the destination — local customs, etiquette,
    or a quirky insight. Keep it brief and practical.
    """,
)


@tool
async def search_activities(runtime: ToolRuntime[None, TripState]) -> str:
    """Activity team-leader: get weather, activities, and culture tips for the destination."""
    destination = runtime.state["destination"]

    # Try Open-Meteo first, fall back to Tavily if it fails
    weather_result = ""
    try:
        response = await openmeteo_executor.ainvoke({
            "messages": [HumanMessage(content=f"Weather forecast for {destination}")]
        })
        weather_result = response["messages"][-1].content
        if "error" in weather_result.lower() or "failed" in weather_result.lower():
            raise ValueError("Open-Meteo returned an error")
    except Exception:
        response = await tavily_weather_executor.ainvoke({
            "messages": [HumanMessage(content=f"Weather forecast for {destination}")]
        })
        weather_result = response["messages"][-1].content

    activities_response = await activities_executor.ainvoke({
        "messages": [HumanMessage(content=f"Cheap things to do in {destination}")]
    })

    culture_response = culture_executor.invoke({
        "messages": [HumanMessage(content=f"Cultural tips for visiting {destination}")]
    })

    return "\n\n---\n\n".join([
        f"Weather:\n{weather_result}",
        f"Activities:\n{activities_response['messages'][-1].content}",
        f"Culture:\n{culture_response['messages'][-1].content}",
    ])
