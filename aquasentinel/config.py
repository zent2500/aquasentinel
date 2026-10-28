"""런타임 설정: 모델 선택(Gemini / Offline) + MCP 연결(stdio / Streamable HTTP)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_NAME = os.getenv("ADK_MODEL_NAME", "gemini-2.5-flash")
OFFLINE = os.getenv("AQUA_OFFLINE", "0") == "1"
MCP_URL = os.getenv("AQUA_MCP_URL", "").strip()
SERVER_PATH = Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"


def model_for(role: str):
    """역할별 모델. 오프라인 모드면 결정론 스크립트 LLM, 아니면 Gemini 모델명."""
    if OFFLINE:
        from .offline_llm import OfflineScriptedLlm
        return OfflineScriptedLlm(model="offline/aquasentinel", role=role)
    return MODEL_NAME


def mcp_toolset(tool_filter: list[str]):
    """FastMCP 서버 연결. AQUA_MCP_URL 이 있으면 HTTP(Cloud Run), 없으면 stdio 자식 프로세스."""
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import (
        StdioConnectionParams, StreamableHTTPConnectionParams)
    from mcp import StdioServerParameters

    if MCP_URL:
        params = StreamableHTTPConnectionParams(url=MCP_URL, timeout=30)
    else:
        params = StdioConnectionParams(
            server_params=StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)]),
            timeout=30,
        )
    return McpToolset(connection_params=params, tool_filter=tool_filter)
