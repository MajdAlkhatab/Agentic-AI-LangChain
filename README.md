# My LangChain Draft Notes

**System prompt:** is used to change the behavior of the model, e.g. using few shot learning or just using instructions. 

**create_agent:** creates agent built on top of LangGraph, with more features than standard chatbot, such as memory, tools, step tracking and state. 

**response_format:** You can specify the exact structure of the model output (return object, schema).

**Tool:** you can create your custom function (tools) that the model can use when needed. The agent reads what inside @tool such as function name, description and expected input.

**Tavily:** powerful search tool extension to LLMs

**Memory:** Is the controversial history. You pass a config dictionary (like {"configurable": {"thread_id": "1"}}) to tell the checkpointer exactly which saved conversation session to load and update. 

**Input type:** It can be text, image or audio. Each one has its own way to input. LLM expects Base64 Encoding format for image and audio.

**MCP:** connect LLM not only to external tools, but also context (resources) you. you can extend LangChain's agent with @tool, @mcp.tool(), @mcp.resource() or  @mcp.prompt() as MCP standard.

It is better to use await when communicating with external systems like servers or APIs because it forces Python to pause and actually receive the response before running the next line of code. 

**Context Injection:** you can inject the data to the Agent (but not the LLM) without using prompts by using context_schema class objects. And you can ingest the data without prompts to the agent and the LLM by letting the LLM have access to context_schema

**State:** Works as a shared notepad where the LLM can read/update past notes, and write new notes to remember across your entire conversation thread. To change the notepad you give the LLM tools to do so. 

**Subagent:** You can define a subagent as a tool, so your Main Agent can call mini subagents to do a specific task they are experts at. Each subagent is a tool (inside each tool there is a highly specialized subagent). Only the output of the subagent is stored in the chat history.

**Memory Management:** To handle long chats, save tokens, and prevent hallucinations, you can use Summarization (compressing the old chat history into a short summary once it hits a token limit, keeping only the most recent prompt raw) or Message Trimming (it uses a before_agent or similar hook with RemoveMessage to delete messages from state) or Custom Cleaning (where you define a custom function the clean the output/logs of a tool call).

**Dynamic Model Routing:** automatically switching between different AI models based on some condition such as the conversation's length, to save costs and handle context window limits. 

**LanguageContext:** A setting used to force the agent to output its responses in a specific language.

**Dynamic Tool Access (Role-Based Access Control):** It restricts which tools the AI can use based on who is talking to it.  

**Middleware:** is an extra step between the user prompt and the agent's response. For example, you can add an additional step before or after the prompt goes to the LLM, such as getting human approval to send an email, summarizing a text to save tokens, hiding sensitive personal data before it goes to the agent, or retrying an MCP tool call if it fails.


**Runtime (or Runtime Context):** is a hidden setup that the agent uses to interact with a specific user. Instead of hardcoding a user ID or user Name in your code as a variable, or writing it out in the prompt, you can pass it into the runtime context (like a .env file in VSCode to store keys and variables invisibly). This way, the Agent and its tools have access to it behind the scenes without cluttering the prompt or the code. The variables are defined as context schema and stored in a place called runtime (inside the state). it gives your custom tools live access to the application's background memory (Variables, Context and State) behind the scenes while they are executing. 

**state:** is a memory that stores the conversion history and custom fields (variables with values).


**MCP Server Directories**:
    * [Awesome MCP Servers (GitHub)](https://github.com/punkpeye/awesome-mcp-servers)
    * [Glama MCP Servers](https://glama.ai/mcp/servers)
    * [MCP.so](https://mcp.so/)


