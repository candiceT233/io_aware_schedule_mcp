#!/bin/bash

ITERATIONS=10

echo "📦 Syncing dependencies..."
uv sync

if [ -f "benchmark_results.csv" ]; then
    echo "🗑️  Removing old benchmark_results.csv..."
    rm benchmark_results.csv
fi

# Round 1: Stateless
echo ""
echo "-----------------------------------------------------"
echo "🚀 Round 1: STATELESS Mode ($ITERATIONS Iterations)"
echo "-----------------------------------------------------"
CONTEXT_MODE=stateless uv run MCP_Client/client_benchmark.py MCP_Server/main.py --auto --iterations $ITERATIONS

# Round 2: Session
echo ""
echo "-----------------------------------------------------"
echo "🚀 Round 2: SESSION Mode ($ITERATIONS Iterations)"
echo "-----------------------------------------------------"
CONTEXT_MODE=session uv run MCP_Client/client_benchmark.py MCP_Server/main.py --auto --iterations $ITERATIONS

echo ""
echo "✅ Benchmark Complete!"


# Round 3: Persistent
echo ""
echo "-----------------------------------------------------"
echo "🚀 Round 3: PERSISTENT Mode (Redis Backend)"
echo "-----------------------------------------------------"

nc -z localhost 6379 || echo "⚠️ WARNING: Redis might not be running on localhost:6379!"
export REDIS_HOST=localhost
export REDIS_PORT=6379

CONTEXT_MODE=persistent uv run MCP_Client/client_benchmark.py MCP_Server/main.py --auto --iterations $ITERATIONS

echo ""
echo "✅ All Benchmarks Complete!"
echo "📊 Results saved to: $(pwd)/benchmark_results.csv"