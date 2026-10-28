"""AuditTrailPlugin — ADK Plugin 으로 전 에이전트·도구 실행을 감사 로그(JSONL)로 남긴다.

에이전트마다 콜백을 붙이지 않고 Runner(App) 레벨에서 한 번에 관측성을 확보하는 방식.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

LOG_PATH = Path(__file__).resolve().parent.parent / "outputs" / "audit_trail.jsonl"


class AuditTrailPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="audit_trail")
        self._t0: dict[str, float] = {}

    def _write(self, rec: dict[str, Any]) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    async def before_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> Optional[Any]:
        self._t0[agent.name] = time.perf_counter()
        self._write({"event": "agent_start", "agent": agent.name, "invocation": callback_context.invocation_id})
        return None

    async def after_agent_callback(self, *, agent: BaseAgent, callback_context: CallbackContext) -> Optional[Any]:
        dt = time.perf_counter() - self._t0.pop(agent.name, time.perf_counter())
        self._write({"event": "agent_end", "agent": agent.name, "elapsed_s": round(dt, 3)})
        return None

    async def after_tool_callback(self, *, tool: BaseTool, tool_args: dict[str, Any],
                                  tool_context: ToolContext, result: dict) -> Optional[dict]:
        status = result.get("status") if isinstance(result, dict) else None
        self._write({"event": "tool", "agent": tool_context.agent_name, "tool": tool.name,
                     "args": tool_args, "status": status})
        return None
