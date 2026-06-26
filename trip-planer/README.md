# Agentic AI Trip Planner

The best way to learn is by doing. After finishing up my LangChain learning, I wanted to build a project I've always wished existed: an AI agent that runs automatically every morning, searches for travel opportunities, and plans a whole trip for me. So that if it finds a plan and price that is an opportunity truly worth buying, I'll take it. Doing this manually is a really boring task to repeat every day. By automating the process, it saves me lots of hours of doing boring tasks. So, I built an AI agent to do that boring work.

### How it works

The model is simple. We have one main agent and multiple sub-agents. The sub-agents are experts in narrow things, while the main agent orchestrates the work for them. I built 6 sub-agents in this workflow:

1. **Travel Agent:** Searches for flights (uses the Kiwi MCP).
2. **Hotel Agent:** Searches for hotels (uses the Kiwi MCP).
3. **Playlist Agent:** An SQL expert that creates an in-flight playlist from a local database.
4. **Activities Agent:** Searches for activities (uses Tavily search).
5. **Weather Agent:** Looks up the weather forecast (uses Tavily search).
6. **Culture Agent:** Provides quirky recommendations (uses its internal knowledge).

All of these agents are orchestrated by a **Main Agent**, which gathers the constraints, delegates the tasks, and compiles the final itinerary.


### How to use it

To run it, you only need to install the dependencies and provide your API keys.

**Requirements:**
* A [Tavily API Key](https://tavily.com/) for web search capabilities.
* An [OpenAI API Key](https://platform.openai.com/) (defaults to `gpt-4o-mini`).

**1. Create a virtual environment**

Mac/Linux:
```bash
python -m venv venv && source venv/bin/activate
```
Windows (Git Bash):
```bash
python -m venv venv && source venv/Scripts/activate
```

**2. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**3. Set up your environment variables**

Copy `.env.example` to `.env` and add your keys:

Mac/Linux:
```bash
cp .env.example .env
```
Windows:
```bash
copy .env.example .env
```
```text
OPENAI_API_KEY='*****'
TAVILY_API_KEY='*****'
```

**4. Run the agent**
```bash
python trip_planner.py
```

The agent will read the prompt inside the script, delegate the research to the sub-agents, and print a complete, budget-conscious travel plan to your terminal.
You can deploy it for free on Railway or GitHub Actions, and receive the plan as a Telegram message.