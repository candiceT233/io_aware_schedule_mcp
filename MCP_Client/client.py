import asyncio
import sys
import json
import time
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack

# Imports for MCP (Modular Chat Protocol)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Imports for Gemini API (We use standard libraries for HTTP requests)
import aiohttp
from dotenv import load_dotenv

import os

load_dotenv()


API_KEY = os.getenv("API_KEY")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"
# ------------------------------------

class MCPClient:
    """
    Client utilizing the Gemini model for tool use via the Modular Chat Protocol (MCP).
    """
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.chat_history: List[Dict[str, Any]] = []

    # --- Utility for Robust API Calls ---
    async def _gemini_api_call(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handles the HTTP POST request to the Gemini API with exponential backoff."""
        max_retries = 5
        base_delay = 1.0
        
        headers = {
            'Content-Type': 'application/json',
            # Authorization headers are often handled by the runtime or implicit environment setup
        }

        async with aiohttp.ClientSession() as http_session:
            for attempt in range(max_retries):
                try:
                    # Append API key to URL if present
                    url_with_key = API_URL
                    if API_KEY:
                        url_with_key += f"?key={API_KEY}"
                    
                    async with http_session.post(url_with_key, json=payload, headers=headers) as response:
                        response_json = await response.json()
                        
                        if response.status == 200:
                            return response_json
                        
                        # Handle specific API errors
                        elif response.status in (429, 500, 503):
                            print(f"API Error (Status {response.status}). Retrying in {base_delay * (2 ** attempt):.2f}s...")
                            await asyncio.sleep(base_delay * (2 ** attempt))
                            continue
                        else:
                            print(f"Unexpected API Error (Status {response.status}): {response_json}")
                            return None

                except aiohttp.ClientError as e:
                    print(f"Network error during API call: {e}. Retrying...")
                    await asyncio.sleep(base_delay * (2 ** attempt))
                
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    return None
            
            print("Failed to call Gemini API after multiple retries.")
            return None
    # ------------------------------------

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server (Logic remains the same)"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools and convert to Gemini format
        response = await self.session.list_tools()
        self.available_tools = []
        for tool in response.tools:
            # MCP's inputSchema uses JSON Schema, which is compatible with Gemini's function declaration format
            self.available_tools.append({
                "functionDeclarations": [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema # This is the key mapping
                }]
            })
        
        tool_names = [t['functionDeclarations'][0]['name'] for t in self.available_tools]
        print(f"\nConnected to server with tools: {tool_names}")


    async def process_query(self, query: str) -> str:
        """Process a query using Gemini and available tools (Updated Logic)"""
        
        # 1. Start or update chat history
        self.chat_history.append({"role": "user", "parts": [{"text": query}]})
        
        final_text = []
        tool_call_count = 0
        
        while True:
            # 2. Construct Gemini payload
            payload = {
                "contents": self.chat_history,
                "generationConfig": { # <--- FIX: Changed 'config' to 'generationConfig'
                    "temperature": 0.0
                }
            }
            if self.available_tools:
                 # <--- FIX: Moved 'tools' to be a top-level field
                payload["tools"] = self.available_tools

            # 3. Call the Gemini API
            api_response = await self._gemini_api_call(payload)

            if not api_response:
                return "Error: Could not get a response from the Gemini API."

            candidate = api_response.get('candidates', [{}])[0]
            if not candidate or 'content' not in candidate:
                 # If model blocks the response or returns empty content
                return "Error: Received empty or unsafe response from the model."
            
            response_content = candidate['content']['parts']
            
            # Reset parts for the assistant's turn in history
            assistant_parts = []
            
            # Check for tool use
            tool_calls = [part['functionCall'] for part in response_content if 'functionCall' in part]
            
            if tool_calls:
                tool_call_count += 1
                
                # A. Prepare the assistant's response parts (which contain the tool_call info)
                # This must be done to keep the chat history correct for the next turn.
                assistant_parts = response_content

                # B. Execute all tool calls
                tool_results_parts = []
                for tool_call in tool_calls:
                    tool_name = tool_call['name']
                    tool_args = dict(tool_call['args'])
                    
                    print(f"--- Tool Use Required: {tool_name}({json.dumps(tool_args)}) ---")
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    
                    # Call the actual tool via MCP session
                    try:
                        mcp_result = await self.session.call_tool(tool_name, tool_args)
                        tool_output = str(mcp_result.content)
                    except Exception as e:
                        tool_output = f"Tool execution error: {e}"
                        print(f"MCP Call Error: {e}")
                    
                    # C. Construct the tool result part for the history
                    tool_results_parts.append({
                        "functionResponse": {
                            "name": tool_name,
                            "response": {"result": tool_output} # Gemini expects the tool output here
                        }
                    })

                # D. Append assistant turn (tool calls) and user turn (tool results) to history
                self.chat_history.append({"role": "model", "parts": assistant_parts})
                self.chat_history.append({"role": "tool", "parts": tool_results_parts})
                
                # E. Loop again to send the tool results back to the model
                # This makes the process iterative until a final text response is received.
                continue

            else:
                # No tool calls, only final text response
                
                # A. Extract final text
                final_text_content = next((part.get('text') for part in response_content if 'text' in part), "")
                
                if final_text_content:
                    final_text.append(final_text_content)
                    
                    # B. Append the final model response to history
                    self.chat_history.append({"role": "model", "parts": response_content})
                
                # C. Break loop and return combined text
                break
                
        return "\n".join(final_text)


    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nGemini-based MCP Client Started!")
        print(f"Model: {MODEL_NAME}")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break
                
                # Reset chat history for a new conversation or handle history management as needed
                # If you want continuous chat memory, you might not want to reset here.
                # For simplicity, we keep history until 'quit' or error.

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                # Print error details but don't stop the loop immediately
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging

    async def cleanup(self):
        """Clean up resources"""
        if self.exit_stack:
            await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_mcp_client.py <path_to_server_script>")
        # The standard Python entrypoint is often main.py, so we can suggest that as a default
        print("Example: python gemini_mcp_client.py ../MCP_Server/main.py") 
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    except Exception as e:
        print(f"\nFatal Error during connection or execution: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    # Add aiohttp to the imports
    if 'aiohttp' not in sys.modules:
        print("Please ensure 'aiohttp' is installed for asynchronous HTTP requests.")
        sys.exit(1)
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient terminated by user.")
    except Exception as e:
        print(f"A non-fatal error occurred in the async loop: {e}")
        sys.exit(1)