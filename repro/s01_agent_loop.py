#!/usr/bin/env python3
"""
Repro for s01: The Agent Loop

这个版本适配你当前的环境：
- Provider: 阿里云百炼
- Protocol: OpenAI compatible API
- Model: MiniMax-M2.5

学习重点不变：
1. 维护 messages 历史
2. 让模型决定是否调用工具
3. 执行工具
4. 把工具结果回填给模型
5. 循环直到模型停止调用工具
"""

from __future__ import annotations

import json
import os
import subprocess

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv(override=True)

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
MODEL = os.getenv("MODEL_ID", "MiniMax-M2.5")

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to solve tasks. Act, don't explain."
)

# OpenAI-compatible tools use the "function" format.
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
    }
]


def run_bash(command: str) -> str:
    """Execute one shell command and return its output."""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
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


def build_assistant_message(message) -> dict:
    """
    Convert SDK response objects into plain message dicts.

    Why this exists:
    - We only want to store the fields needed for the next round.
    - For tool calls, the assistant message must contain tool_calls.
    - For normal answers, content is enough.
    """
    assistant_message = {"role": "assistant"}

    tool_calls = message.tool_calls or []
    if tool_calls:
        assistant_message["content"] = message.content or ""
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
    else:
        assistant_message["content"] = message.content or ""

    return assistant_message


def agent_loop(messages: list[dict]) -> None:
    while True:
        # OpenAI-compatible chat completions puts the system prompt
        # inside messages rather than a separate top-level field.
        request_messages = [{"role": "system", "content": SYSTEM}, *messages]

        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        # Always append the full assistant turn so the next round can
        # see whether the model responded with plain text or tool_calls.
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
                if tool_name != "bash":
                    output = f"Error: Unknown tool: {tool_name}"
                else:
                    command = arguments.get("command", "")
                    if not command:
                        output = "Error: Missing command"
                    else:
                        print(f"\033[33m$ {command}\033[0m")
                        output = run_bash(command)
                        print(output[:200])

            # In OpenAI-compatible tool calling, tool results come back
            # as role="tool" messages tied to a specific tool_call_id.
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
            query = input("\033[36mrepro-s01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        last = history[-1]
        if last.get("role") == "assistant" and last.get("content"):
            print(last["content"])

        print()
