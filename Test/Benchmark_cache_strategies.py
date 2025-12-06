import time
import asyncio
import os
import sys
from tabulate import tabulate

# --- 路径配置 (Path Configuration) ---
# 获取当前脚本 (benchmark_strategies.py) 所在的目录 (Test/)
CURRENT_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# 计算 MCP_Server 的路径 (Test/../MCP_Server)
MCP_SERVER_DIR = os.path.join(CURRENT_TEST_DIR, '..', 'MCP_Server')
MCP_SERVER_DIR = os.path.abspath(MCP_SERVER_DIR)

# 将 MCP_Server 添加到 sys.path 以便导入 main.py
if MCP_SERVER_DIR not in sys.path:
    sys.path.append(MCP_SERVER_DIR)

# 同时也需要确保 workflow_analysis 能够被正确找到 (如果 main.py 依赖它)
# 通常 main.py 内部会处理它自己的依赖路径，但为了保险起见，
# 我们也可以把 MCP_Server/spm/workflow_analysis 加进去，或者依赖 main.py 的处理
# 这里我们假设 main.py 能够处理好它自己的子模块路径。

print(f"Added server path: {MCP_SERVER_DIR}")

try:
    # 从 main.py 导入工具函数
    from main import recommend_single_task_storage
except ImportError as e:
    print(f"Error importing main.py: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

async def run_benchmark():
    print("=== RQ4: Context Utilization & Caching Strategy Benchmark ===")
    
    # 基础参数 (Base Payload)
    base_payload = {
        "file_size_mb": 102400,
        "operation": "write",
        "transfer_size_bytes": 4194304,
        "parallelism": 32,
        "num_nodes": 4,
        "needs_persistence": True,
        "task_name": "heavy_sim"
    }

    results = []

    # --- 场景 1: Stateless Mode (无状态模式) ---
    # 每次请求必须包含所有参数。模拟传统 REST API。
    print("\n>>> Scenario 1: Stateless (Always sending full context)")
    
    # Request 1
    start = time.perf_counter()
    await recommend_single_task_storage(**base_payload, session_id="session_A")
    t1 = time.perf_counter() - start
    
    # Request 2 (假设用户想改参数，必须重发所有)
    payload_v2 = base_payload.copy()
    payload_v2['operation'] = 'read'
    start = time.perf_counter()
    await recommend_single_task_storage(**payload_v2, session_id="session_A_new") # 强制新 Session 模拟无状态
    t2 = time.perf_counter() - start
    
    results.append(["Stateless", "Full Payload x2", f"{(t1+t2)*1000:.2f} ms", "Low Context Utilization"])

    # --- 场景 2: Stateful / Session-based (有状态模式) ---
    # 利用 MCP 的上下文保持能力。
    print("\n>>> Scenario 2: Stateful (Partial updates)")
    
    # Request 1: 初始化上下文 (Full Payload)
    start = time.perf_counter()
    await recommend_single_task_storage(**base_payload, session_id="session_B")
    t3 = time.perf_counter() - start
    
    # Request 2: 增量更新 (Only Delta)
    # LLM 只需要发送改变的参数："其实我是要读操作"
    delta_payload = {"operation": "read"} 
    start = time.perf_counter()
    # 注意：这里我们没有传 file_size 等参数，服务器会自动从 session_B 获取
    res = await recommend_single_task_storage(**delta_payload, session_id="session_B")
    t4 = time.perf_counter() - start
    
    # 验证上下文是否生效
    throughput_val = res.get('predicted_throughput', '0.00')
    # 处理可能的非字符串返回 (虽然预期是字符串)
    if isinstance(throughput_val, str):
        is_valid = throughput_val != '0.00 MiB/s' and res.get('best_option')
    else:
        is_valid = throughput_val > 0 and res.get('best_option')

    status = "Success" if is_valid else "Failed"
    
    results.append(["Stateful", "Full + Delta", f"{(t3+t4)*1000:.2f} ms", f"High Utilization ({status})"])

    # --- 场景 3: Cache Hit (纯缓存) ---
    # 完全相同的请求
    print("\n>>> Scenario 3: Memoization (Cache Hit)")
    start = time.perf_counter()
    await recommend_single_task_storage(**delta_payload, session_id="session_B")
    t5 = time.perf_counter() - start
    results.append(["Memoization", "Delta (Cached)", f"{t5*1000:.2f} ms", "Instant Response"])

    # --- 报告 ---
    print("\n" + "="*65)
    print("RQ4 RESULTS: ARCHITECTURE COMPARISON")
    print("="*65)
    print(tabulate(results, headers=["Architecture", "Input Data", "Total Latency", "Context Efficiency"], tablefmt="grid"))
    
    print("\n[Analysis]")
    print("1. Stateless 模式要求客户端每次维护完整状态，数据传输量大。")
    print("2. Stateful 模式允许 'Delta' 更新，服务器接管上下文记忆 (Memory Offloading)。")
    print("3. Memoization 进一步消除了重复计算的开销。")

if __name__ == "__main__":
    asyncio.run(run_benchmark())