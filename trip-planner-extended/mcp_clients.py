import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from typing import Dict, Any
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent

load_dotenv()

RETRYABLE_MCP_CODES = {-32603}


class RetryMCPInterceptor:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


async def init_mcp_tools():
    client = MultiServerMCPClient(
        {
            "skiplagged": {
                "transport": "streamable_http",
                "url": "https://mcp.skiplagged.com/mcp",
            },
            "kiwi": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com",
            },
            "trivago": {
                "transport": "streamable_http",
                "url": "https://mcp.trivago.com/mcp",
            },
            "frankfurter": {
                "transport": "streamable_http",
                "url": "https://mcp.frankfurter.dev/",
            },
            "airbnb": {
                "transport": "stdio",
                "command": "mcp-server-airbnb",
                "args": ["--ignore-robots-txt"],
            },
            "hotelzero": {
                "transport": "stdio",
                "command": "hotelzero",
                "args": [],
            },
            "weather": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "mcp_weather_server"],
            },
        },
        tool_interceptors=[RetryMCPInterceptor()],
    )
    all_tools = await client.get_tools()
    return all_tools


# --- Shared Tavily web search tool ---
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}
