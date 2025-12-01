import asyncio
import sys
import json
import os
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

# Imports for MCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Imports for HTTP/API
import aiohttp
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# 系统提示词：告诉 LLM 如何使用复杂的存储推荐工具
SYSTEM_INSTRUCTION = """
You are an expert HPC (High Performance Computing) Storage Consultant. 
Your goal is to help scientists and engineers optimize their I/O workflows.

You have access to a tool named 'recommend_storage' which uses advanced interpolation on IOR benchmarks.
To give an accurate recommendation, you need to gather specific details about the user's workload.

Do not guess parameters if possible. Ask the user for clarification if the following are missing:
1. Operation type (read vs write)
2. Total file size
3. Transfer block size (Transfer Size) - Crucial for I/O performance
4. Scale of the job:
   - How many compute nodes? (num_nodes)
   - How many tasks/processes per node? (tasks_per_node)
   
If the user provides a simple request (e.g., "I want to write 10GB"), assume defaults (1 node, 1 task, 1MB transfer size) but inform them you are using defaults.
"""

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.chat_history: List[Dict[str, Any]] = []
        self.available_tools_schema: List[Dict[str, Any]] = []

    async def _gemini_api_call(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handles HTTP POST to Gemini API."""
        headers = {'Content-Type': 'application/json'}
        url = f"{API_URL}?key={API_KEY}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        text = await response.text()
                        print(f"API Error {response.status}: {text}")
                        return None
                    return await response.json()
            except Exception as e:
                print(f"Network Error: {e}")
                return None

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server and parse tools for Gemini."""
        print(f"Connecting to server: {server_script_path}...")
        
        server_params = StdioServerParameters(
            command="python", # 假设都是 Python 服务
            args=[server_script_path],
            env=os.environ.copy() # 传递环境变量以防 server 需要
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
        await self.session.initialize()

        # Fetch tools
        response = await self.session.list_tools()
        
        # --- Critical Fix: Correct Gemini Schema Structure ---
        # Gemini expects: tools = [{ "function_declarations": [ func1, func2... ] }]
        funcs = []
        for tool in response.tools:
            funcs.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema # MCP Schema is compatible
            })
        
        if funcs:
            self.available_tools_schema = [{
                "function_declarations": funcs
            }]
            
        tool_names = [t['name'] for t in funcs]
        print(f"Connected! Available tools: {', '.join(tool_names)}")

    async def process_query(self, query: str) -> str:
        # 1. Add user message
        self.chat_history.append({"role": "user", "parts": [{"text": query}]})
        
        while True:
            # 2. Build Payload
            payload = {
                "contents": self.chat_history,
                "system_instruction": {
                    "parts": [{"text": SYSTEM_INSTRUCTION}]
                },
                "tools": self.available_tools_schema,
                "generationConfig": {"temperature": 0.0} # Low temp for precise tool use
            }

            # 3. Call API
            response = await self._gemini_api_call(payload)
            if not response: return "API Call Failed."

            # 4. Parse Response
            try:
                candidate = response['candidates'][0]['content']
                model_parts = candidate.get('parts', [])
            except (KeyError, IndexError):
                return "Error: Invalid response format from API."

            # 5. Check for Function Calls
            function_calls = []
            final_text_content = ""

            for part in model_parts:
                if 'functionCall' in part:
                    function_calls.append(part['functionCall'])
                if 'text' in part:
                    final_text_content += part['text']

            # Case A: Pure Text Response -> Done
            if not function_calls:
                self.chat_history.append(candidate) # Save model turn
                return final_text_content

            # Case B: Tool Use Required
            print(f"\n[AI is thinking...] invoking {len(function_calls)} tools.")
            self.chat_history.append(candidate) # Save model turn (with function calls)

            # 6. Execute Tools
            function_responses = []
            for call in function_calls:
                name = call['name']
                args = call['args']
                print(f"  > Calling {name} with args: {json.dumps(args, ensure_ascii=False)}")

                try:
                    # MCP Execution
                    result = await self.session.call_tool(name, arguments=args)
                    # Convert MCP result to string for LLM
                    tool_output = result.content[0].text if result.content else "Success"
                except Exception as e:
                    tool_output = f"Error: {str(e)}"

                # Construct Response Part (Gemini REST format)
                function_responses.append({
                    "functionResponse": {
                        "name": name,
                        "response": {"content": tool_output} 
                    }
                })

            # 7. Append Function Results to History
            self.chat_history.append({
                "role": "user", 
                "parts": function_responses
            })
            
            # Loop back to send results to model

    async def chat_loop(self):
        print("\n--- Storage Recommender Client (Gemini + MCP) ---")
        print("Tip: Try asking 'I run a 4-node simulation writing 50GB data per task...'")
        
        while True:
            try:
                q = input("\nUser: ").strip()
                if q.lower() in ('quit', 'exit'): break
                if not q: continue
                
                resp = await self.process_query(q)
                print(f"\nAI: {resp}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        if self.exit_stack: await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server.py>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())