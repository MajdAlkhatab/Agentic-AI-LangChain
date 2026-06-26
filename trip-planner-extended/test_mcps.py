import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()


def print_tools(name: str, tools: list):
    print(f"\n{'='*50}")
    print(f"MCP: {name} ({len(tools)} tools)")
    print('='*50)
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")


async def test_http_mcp(name: str, url: str):
    try:
        client = MultiServerMCPClient({name: {"transport": "streamable_http", "url": url}})
        tools = await client.get_tools()
        print_tools(name, tools)
    except Exception as e:
        print(f"\n[FAILED] {name}: {e}")


async def test_stdio_mcp(name: str, command: str, args: list):
    try:
        client = MultiServerMCPClient({name: {"transport": "stdio", "command": command, "args": args}})
        tools = await client.get_tools()
        print_tools(name, tools)
    except Exception as e:
        print(f"\n[FAILED] {name}: {e}")


async def main():
    # --- HTTP MCPs (no subprocess, no Windows fix needed) ---
    await test_http_mcp("frankfurter",  "https://mcp.frankfurter.dev/")
    await test_http_mcp("skiplagged",   "https://mcp.skiplagged.com/mcp")
    await test_http_mcp("kiwi",         "https://mcp.kiwi.com")
    await test_http_mcp("trivago",      "https://mcp.trivago.com/mcp")

    # --- stdio MCPs (subprocess-based, requires ProactorEventLoop on Windows) ---
    await test_stdio_mcp("airbnb",   "mcp-server-airbnb", ["--ignore-robots-txt"])
    await test_stdio_mcp("hotelzero","hotelzero", [])
    await test_stdio_mcp("weather",  "python", ["-m", "mcp_weather_server"])


if __name__ == "__main__":
    asyncio.run(main())