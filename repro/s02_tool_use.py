#!/usr/bin/env python3
"""
Repro for s02: Tool Use

这个版本适配：
- Provider: 阿里云百炼
- Protocol: OpenAI compatible API
- Model: MiniMax-M2.5

s02 的学习重点：
1. s01 的 agent loop 不应该大改
2. 新增工具时，核心是新增 tool schema + handler
3. 用 TOOL_HANDLERS 做 dispatch map
4. 文件工具要先经过 safe_path，避免路径逃逸
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

WORKDIR = Path.cwd()

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
MODEL = os.getenv("MODEL_ID", "MiniMax-M2.5")

SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Use tools to solve tasks. Act, don't explain."
)

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path



def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"





def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"



def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# OpenAI-compatible tool schema.
# 它们是给模型看的“工具说明书”，不是工具实现本身。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


def build_assistant_message(message) -> dict:
    """Convert an OpenAI SDK message into a plain dict for messages."""
    assistant_message = {
        "role": "assistant",
        "content": message.content or "",
    }

    tool_calls = message.tool_calls or []
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in tool_calls
        ]

    return assistant_message


def dispatch_tool(tool_name: str, arguments: dict) -> str:
    """Find the handler for tool_name and execute it with arguments."""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool: {tool_name}"

    try:
        return str(handler(**arguments))
    except Exception as e:
        return f"Error: {e}"


def agent_loop(messages: list[dict]) -> None:
    """Run the agent loop until the model stops requesting tools."""
    while True:
        request_messages = [{"role": "system", "content": SYSTEM}, *messages]
        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        messages.append(build_assistant_message(message))

        tool_calls = message.tool_calls or []
        if not tool_calls:
            return

        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                output = dispatch_tool(tool_name, arguments)

            print(f"> {tool_name}: {output[:200]}")
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": output,
                }
            )

        messages.extend(results)


if __name__ == "__main__":
    history: list[dict] = []

    while True:
        try:
            query = input("\033[36mrepro-s02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        last = history[-1]
        if isinstance(last, dict) and last.get("role") == "assistant" and last.get("content"):
            print(last["content"])

        print()
