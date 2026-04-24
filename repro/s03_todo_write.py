#!/usr/bin/env python3
"""
Repro for s03: TodoWrite

This follows the original s03 lesson closely:
1. Add a TodoManager as structured planning state
2. Expose that state through a todo tool
3. Remind the model to update todos when it forgets

The only provider-specific change is the OpenAI-compatible tool-calling format.
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
    "Use the todo tool to plan multi-step tasks. "
    "Mark in_progress before starting, completed when done. "
    "Prefer tools over prose."
)


class TodoManager:
    def __init__(self):
        # 在内存里保存当前 todo 列表。
        self.items: list[dict] = []

    def update(self, items: list) -> str:
        # 限制 todo 数量，避免模型一次维护过多任务。
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        # 不直接相信模型传来的原始数据，先构造一份校验后的副本。
        validated = []
        # 只允许一个任务处于进行中，强制模型聚焦当前步骤。
        in_progress_count = 0

        for i, item in enumerate(items):
            # 读取并规范化模型传来的每个字段。
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            # 每个 todo 都必须有任务描述。
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            # 状态只能是我们定义好的三种之一。
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            # 统计当前有多少个进行中的任务，后面要做唯一性校验。
            if status == "in_progress":
                in_progress_count += 1

            # 保存规范化后的任务项，而不是直接保存原始输入。
            validated.append({"id": item_id, "text": text, "status": status})

        # 最多只允许一个任务是 in_progress。
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        # 用校验后的新状态替换旧的 todo 列表。
        self.items = validated
        # 返回渲染后的文本，让模型能看到当前计划进度。
        return self.render()

    def render(self) -> str:
        # 如果当前还没有任何 todo，就明确返回空状态。
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            # 把内部状态转成更直观的进度标记。
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item["status"]]
            # 每个任务渲染成一行文本。
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        # 统计已完成任务数量，生成一个简短的进度汇总。
        done = sum(1 for item in self.items if item["status"] == "completed")
        # 在末尾追加类似 "(2/5 completed)" 的进度说明。
        lines.append(f"\n({done}/{len(self.items)} completed)")
        # 返回最终的多行 todo 文本。
        return "\n".join(lines)


TODO = TodoManager()


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
        r = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
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


TOOLS = [
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
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Update task list. Track progress on multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["id", "text", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    },
]


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "todo": lambda **kw: TODO.update(kw["items"]),
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


def agent_loop(messages: list[dict]) -> None:
    rounds_since_todo = 0

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

        if message.content:
            print(message.content)

        tool_calls = message.tool_calls or []
        if not tool_calls:
            return

        used_todo = False
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name

            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                output = dispatch_tool(tool_name, arguments)

            print(f"> {tool_name}:")
            print(output[:200])

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": output,
                }
            )

            if tool_name == "todo":
                used_todo = True

        messages.extend(results)

        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            messages.append(
                {
                    "role": "user",
                    "content": "<reminder>Update your todos.</reminder>",
                }
            )


if __name__ == "__main__":
    history: list[dict] = []

    while True:
        try:
            query = input("\033[36mrepro-s03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break

        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()
