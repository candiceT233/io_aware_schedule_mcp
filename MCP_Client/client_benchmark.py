import asyncio
import sys
import json
import os
import time
import csv
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import aiohttp
from dotenv import load_dotenv

import argparse

load_dotenv()

# --- Configuration ---
API_KEY = os.getenv("API_KEY")
MODEL_NAME = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# --- Benchmark Data Structure ---
@dataclass
class BenchmarkResult:
    iteration: int 
    query_id: int
    query_text: str
    mode: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls_count: int

class BenchmarkLogger:
    def __init__(self, filename="benchmark_results.csv"):
        self.filename = filename
        self.headers = ["Timestamp", "Mode", "Iteration", "Query_ID", "Query", 
                        "Latency(s)", "Input_Tokens", "Output_Tokens", "Total_Tokens", "Tool_Calls"]
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)

    def log(self, res: BenchmarkResult):
        with open(self.filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%H:%M:%S"),
                res.mode,
                res.iteration,    
                res.query_id,
                res.query_text,
                f"{res.latency_seconds:.4f}",
                res.input_tokens,
                res.output_tokens,
                res.total_tokens,
                res.tool_calls_count
            ])
        print(f"   [Iter {res.iteration} | Q{res.query_id}] Latency: {res.latency_seconds:.2f}s | InTokens: {res.input_tokens}")

# --- Client Logic ---
SYSTEM_INSTRUCTION = """
You are an HPC Storage Consultant. 
ALWAYS use tools when available.
If parameters are missing, use defaults (1MB size, 1 node).
Provide concise answers.
"""

class MCPClient:
    def __init__(self, mode_name="unknown"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.chat_history: List[Dict[str, Any]] = []
        self.available_tools_schema: List[Dict[str, Any]] = []
        self.mode_name = mode_name
        self.logger = BenchmarkLogger()

    async def _gemini_api_call(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        headers = {'Content-Type': 'application/json'}
        url = f"{API_URL}?key={API_KEY}"
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status != 200: 
                        print(f"API Error: {response.status}")
                        return None
                    return await response.json()
            except Exception as e:
                print(f"Network Error: {e}")
                return None

    async def connect_to_server(self, server_script_path: str):
        print(f"🔌 Connecting to server ({self.mode_name})...")
        # 传递当前环境变量
        env = os.environ.copy()
        
        server_params = StdioServerParameters(
            command="python", 
            args=[server_script_path],
            env=env
        )
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self.exit_stack.enter_async_context(ClientSession(*stdio_transport))
        await self.session.initialize()
        
        # Tool loading logic (simplified)
        response = await self.session.list_tools()
        funcs = [{"name": t.name, "description": t.description, "parameters": t.inputSchema} for t in response.tools]
        if funcs: self.available_tools_schema = [{"function_declarations": funcs}]

    # --- 修复点 1: 增加 iteration 参数 ---
    async def process_query(self, query: str, query_id: int = 0, iteration: int = 1) -> str:
        # Start Timer
        start_time = time.perf_counter()
        
        self.chat_history.append({"role": "user", "parts": [{"text": query}]})
        
        tool_call_count = 0
        final_text = ""
        last_usage = {}

        while True:
            payload = {
                "contents": self.chat_history,
                "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
                "tools": self.available_tools_schema,
                "generationConfig": {"temperature": 0.0}
            }

            response = await self._gemini_api_call(payload)
            if not response: return "API Error"

            # Capture Usage
            if 'usageMetadata' in response:
                last_usage = response['usageMetadata']

            try:
                candidate = response['candidates'][0]['content']
                model_parts = candidate.get('parts', [])
            except:
                return "Error parsing"

            function_calls = [p['functionCall'] for p in model_parts if 'functionCall' in p]
            text_parts = [p['text'] for p in model_parts if 'text' in p]
            
            if text_parts: final_text += "".join(text_parts)

            if not function_calls:
                self.chat_history.append(candidate)
                break # Turn complete

            # Execute Tools
            self.chat_history.append(candidate)
            tool_call_count += len(function_calls)
            
            function_responses = []
            for call in function_calls:
                # print(f"  > Calling {call['name']}...") # Optional: debug print
                res = await self.session.call_tool(call['name'], arguments=call['args'])
                function_responses.append({
                    "functionResponse": {
                        "name": call['name'],
                        "response": {"content": res.content[0].text}
                    }
                })
            
            self.chat_history.append({"role": "user", "parts": function_responses})

        # End Timer
        end_time = time.perf_counter()
        
        # --- 修复点 2: 实例化时传入 iteration ---
        result = BenchmarkResult(
            iteration=iteration,
            query_id=query_id,
            query_text=query,
            mode=self.mode_name,
            latency_seconds=end_time - start_time,
            input_tokens=last_usage.get('promptTokenCount', 0),
            output_tokens=last_usage.get('candidatesTokenCount', 0),
            total_tokens=last_usage.get('totalTokenCount', 0),
            tool_calls_count=tool_call_count
        )
        self.logger.log(result)
        
        return final_text

    async def cleanup(self):
        if self.exit_stack: await self.exit_stack.aclose()

# --- Automated Benchmark Runner (Complex Scenario) ---
async def run_auto_benchmark(server_path: str, iterations: int):
    questions = [
        "Analyze a task named 'Sim_Pro_1'. It is a 'write' operation of 51200 MB (50GB). It runs with parallelism=10 across 5 nodes. I need to see all storage rankings.",
        "Now analyze 'Post_Con_1'. It reads the data generated by Sim_Pro_1. Use the same parallelism (10) and nodes (5).",
        "The next step is 'Agg_Pro_2'. It aggregates the data and 'writes' a 5120 MB file. This runs on 1 node with parallelism=2.",
        "Finally, 'Vis_Con_2' 'reads' the aggregated file. It is a single task (parallelism=1) on 1 node.",
        "Wait, can you show me the full list of storage rankings for 'Sim_Pro_1' again? I missed the details."
    ]

    current_mode = os.getenv("CONTEXT_MODE", "unknown")
    print(f"\n🚀 STARTING BENCHMARK | MODE: {current_mode} | ITERATIONS: {iterations}")
    print("-" * 70)

    async with AsyncExitStack() as stack:
        client = MCPClient(mode_name=current_mode)
        client.exit_stack = stack
        await client.connect_to_server(server_path)

        for iter_num in range(1, iterations + 1):
            print(f"\n🔄 Iteration {iter_num}/{iterations}...")
            
            for i, q in enumerate(questions):
                if i > 0: await asyncio.sleep(0.5) 
                
                # --- 修复点 3: 调用时传入 iteration ---
                await client.process_query(q, query_id=i+1, iteration=iter_num)
                
            client.chat_history = [] 
            print(f"   ✅ Iteration {iter_num} complete. History cleared.")

    print("-" * 70)
    print(f"✅ All {iterations} iterations complete.")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("server_path", help="Path to the server script")
    parser.add_argument("--auto", action="store_true", help="Run automated benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Number of times to repeat the test")
    
    args = parser.parse_args()

    if args.auto:
        await run_auto_benchmark(args.server_path, args.iterations)
    else:
        # Interactive mode support
        async with AsyncExitStack() as stack:
            client = MCPClient(mode_name="interactive")
            client.exit_stack = stack
            await client.connect_to_server(args.server_path)
            print("\n--- Interactive Mode (Type 'quit' to exit) ---")
            while True:
                try:
                    q = input("\nUser: ").strip()
                    if q.lower() in ('quit', 'exit'): break
                    if not q: continue
                    
                    resp = await client.process_query(q, query_id=0, iteration=1)
                    print(f"\nAI: {resp}")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())