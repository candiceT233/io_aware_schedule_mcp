#! /bin/bash
source venv/bin/activate
uv sync
uv run MCP_Client/client.py MCP_Server/main.py 