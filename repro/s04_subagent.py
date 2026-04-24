#!/usr/bin/env python3
"""
Repro for s04: Subagents

This follows the original s04 lesson closely:
1. Parent agent gets a task tool
2. task spawns a child agent with fresh messages=[]
3. The child shares the same filesystem but not the parent's conversation
4. Only the child's final summary is returned to the parent

The only provider-specific change is the OpenAI-compatible tool-calling format.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    "Use the task tool to delegate exploration or subtasks."
)
SUBAGENT_SYSTEM = (
    f"You are a coding subagent at {WORKDIR}. "
    "Complete the given task, then summarize your findings."
)


def log(*parts) -> None:
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    location = f"[{Path(__file__).name}:{caller.f_lineno}]" if caller else f"[{Path(__file__).name}]"
    text = " ".join(str(part) for part in parts)
    print(f"{location} {text}")


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
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
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


BASE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
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

CHILD_TOOLS = BASE_TOOLS

PARENT_TOOLS = BASE_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "description": {
                        "type": "string",
                        "description": "Short description of the task",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


def build_assistant_message(message) -> dict:
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
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"

    try:
        return str(handler(**arguments))
    except Exception as e:
        return f"Error: {e}"


def run_subagent(prompt: str) -> str:
    sub_messages: list[dict] = [{"role": "user", "content": prompt}]
    final_message = None

    for _ in range(30):
        request_messages = [{"role": "system", "content": SUBAGENT_SYSTEM}, *sub_messages]
        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=CHILD_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        final_message = message

        sub_messages.append(build_assistant_message(message))

        tool_calls = message.tool_calls or []
        if not tool_calls:
            break

        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                output = dispatch_tool(tool_name, arguments)

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": output[:50000],
                }
            )

        sub_messages.extend(results)

    if final_message and final_message.content:
        return final_message.content
    return "(no summary)"


def agent_loop(messages: list[dict]) -> None:
    while True:
        request_messages = [{"role": "system", "content": SYSTEM}, *messages]
        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=PARENT_TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        messages.append(build_assistant_message(message))

        if message.content:
            log(message.content)

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
                if tool_name == "task":
                    description = arguments.get("description", "subtask")
                    prompt = arguments.get("prompt", "")
                    log(f"> task ({description}): {prompt[:80]}")
                    output = run_subagent(prompt)
                else:
                    output = dispatch_tool(tool_name, arguments)

            log(f"  {output[:200]}")
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
            query = input("\033[36mrepro-s04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)
        log("")
