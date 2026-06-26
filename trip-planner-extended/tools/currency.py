from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage

from state import TripState
from mcp_clients import web_search

frankfurter_executor = None
tavily_currency_executor = None


def init_currency_executors(frankfurter_tools: list):
    global frankfurter_executor, tavily_currency_executor

    frankfurter_executor = create_agent(
        model="gpt-5-nano",
        tools=[t for t in frankfurter_tools if t.name in ("get_rates", "convert")],
        system_prompt="""
        You are a currency specialist. Get the current exchange rate from the user's
        home currency to the destination currency. Return: rate, and what 100 units
        of home currency buys at the destination. No follow-up questions.
        """,
    )

    tavily_currency_executor = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt="""
        You are a currency specialist. Search the web for the current exchange rate
        from the origin currency to the destination currency. Return: rate and a
        practical example. No follow-up questions.
        """,
    )


@tool
async def search_currency(runtime: ToolRuntime[None, TripState]) -> str:
    """Currency team-leader: get the exchange rate for the destination. Falls back to Tavily."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    query = f"Exchange rate from {origin} currency to {destination} currency"

    try:
        response = await frankfurter_executor.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        result = response["messages"][-1].content
        if "error" in result.lower() or "failed" in result.lower():
            raise ValueError("Frankfurter returned an error")
        return result
    except Exception:
        response = await tavily_currency_executor.ainvoke({
            "messages": [HumanMessage(content=query)]
        })
        return response["messages"][-1].content
