#!/usr/bin/env python3
"""
Repro for s08: Background Tasks

后台任务执行演示，遵循原始 s08 课程内容：
1. 在后台线程中运行长时间命令
2. 维护已完成后台任务的通知队列
3. 在每次 LLM 调用前取出队列中的结果并注入回对话

唯一针对特定 provider 的改动是 OpenAI 兼容的工具调用格式。
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import threading
from typing import Any
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 环境变量，override=True 表示覆盖系统已有变量
load_dotenv(override=True)

# 为 Windows 控制台设置 UTF-8 编码，避免中文输出乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# WORKDIR 为当前工作目录，作为安全路径的根目录
WORKDIR = Path.cwd()

# 创建 OpenAI 兼容的客户端，支持 dashscope 等后端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],  # 从环境变量读取 API key
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",  # 默认使用阿里云 dashscope
    ),
)

# 使用的模型 ID，默认从环境变量读取
MODEL = os.getenv("MODEL_ID", "MiniMax-M2.2")

# 系统提示词，告诉模型可以使用 background_run 执行长时间命令
SYSTEM = f"You are a coding agent at {WORKDIR}. Use background_run for long-running commands."


def log(*parts) -> None:
    """
    打印带位置信息的日志，方便调试和追踪代码执行路径。
    位置信息格式：[文件名:行号]
    """
    frame = inspect.currentframe()  # 获取当前帧
    caller = frame.f_back if frame else None  # 获取调用者的帧
    # 构建位置字符串，如果无法获取调用者信息则只显示文件名
    location = f"[{Path(__file__).name}:{caller.f_lineno}]" if caller else f"[{Path(__file__).name}]"
    # 将所有参数转换为字符串并用空格连接
    text = " ".join(str(part) for part in parts)
    print(f"{location} {text}")


# -- BackgroundManager: 后台任务执行 + 通知队列 --
class BackgroundManager:
    """
    后台任务管理器。

    核心功能：
    1. 在后台线程中执行命令（不阻塞主线程）
    2. 维护所有任务的状态和结果
    3. 通过通知队列在 LLM 调用前注入完成通知

    关键概念："即发即忘" - agent 不必在命令运行时阻塞等待。
    """

    def __init__(self):
        # 任务字典：task_id -> {status, result, command}
        self.tasks: dict[str, dict] = {}
        # 通知队列：存储已完成任务的结果，供下次 LLM 调用时注入
        self._notification_queue: list[dict] = []
        # 线程锁：保护共享数据，防止多线程竞争
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """
        启动后台任务，立即返回 task_id。

        关键点：立即返回，不等待命令完成。
        命令在后台线程中异步执行，agent 可以继续处理其他工作。

        Args:
            command: 要执行的命令

        Returns:
            包含 task_id 的确认消息
        """
        # 生成短 UUID (8字符) 作为任务 ID
        task_id = str(uuid.uuid4())[:8]
        # 记录任务初始状态
        self.tasks[task_id] = {
            "status": "running",  # 任务状态：running/completed/timeout/error
            "result": None,        # 任务结果（完成后填充）
            "command": command,     # 原始命令
        }
        # 创建后台线程，daemon=True 确保主程序退出时线程也强制结束
        thread = threading.Thread(
            target=self._execute,   # 线程执行函数
            args=(task_id, command), # 传递给执行函数的参数
            daemon=True,            # 守护线程标志
        )
        thread.start()  # 启动线程（立即返回，不等待）
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str) -> None:
        """
        线程执行函数：在后台线程中运行 subprocess。

        注意：这是在线程中执行的，不阻塞主线程。

        Args:
            task_id: 任务 ID
            command: 要执行的命令
        """
        try:
            # 执行命令，最多等 300 秒（5分钟）
            result = subprocess.run(
                command,
                shell=True,
                cwd=WORKDIR,
                capture_output=True,  # 捕获 stdout 和 stderr
                text=True,           # 返回文本格式
                timeout=300,          # 5分钟超时
            )
            # 合并 stdout 和 stderr，并截断过长输出
            output = (result.stdout + result.stderr).strip()[:50000]
            status = "completed"  # 成功完成
        except subprocess.TimeoutExpired:
            # 超时
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            # 其他错误
            output = f"Error: {e}"
            status = "error"

        # 更新任务状态和结果（主线程可以随时读取）
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"

        # 将完成通知加入队列（供下次 LLM 调用时注入）
        with self._lock:  # 线程锁保护共享数据
            self._notification_queue.append(
                {
                    "task_id": task_id,
                    "status": status,
                    "command": command[:80],      # 截断命令描述
                    "result": (output or "(no output)")[:500],  # 截断结果
                }
            )

    def check(self, task_id: str = None) -> str:
        """
        查看任务状态或列出所有任务。

        Args:
            task_id: 要查看的任务 ID，如果为 None 则列出所有任务

        Returns:
            状态信息字符串
        """
        if task_id:
            # 查看单个任务
            task = self.tasks.get(task_id)
            if not task:
                return f"Error: Unknown task {task_id}"
            return f"[{task['status']}] {task['command'][:60]}\n{task.get('result') or '(running)'}"

        # 列出所有任务
        lines = []
        for tid, task in self.tasks.items():
            lines.append(f"{tid}: [{task['status']}] {task['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list[dict]:
        """
        取出并清空所有待处理的完成通知。

        在每次 LLM 调用前调用，将后台任务的完成结果注入对话。
        这样模型就能知道之前启动的后台任务已经完成。

        Returns:
            通知列表
        """
        with self._lock:  # 线程锁保护
            # 复制通知列表并清空队列
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs


# 全局后台任务管理器实例
BG = BackgroundManager()


# -- 基础工具实现 --

def safe_path(p: str) -> Path:
    """
    安全路径解析，防止通过路径遍历访问工作目录之外的文件。

    防止通过相对路径（如 ../../../etc/passwd）逃离工作目录。
    如果路径越界则抛出 ValueError。
    """
    path = (WORKDIR / p).resolve()  # 解析为绝对路径
    if not path.is_relative_to(WORKDIR):  # 检查是否在 WORKDIR 范围内
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    阻塞式命令执行。

    与 background_run 不同，这里会等待命令完成才返回结果。
    适用于快速命令。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]  # 危险命令黑名单
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,  # 2分钟超时
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    """
    读取文件内容的封装。

    支持行数限制，返回截断后的内容（会标注省略了多少行）。
    最大返回 50000 字符。
    """
    try:
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        # 如果指定了限制且行数超过限制，则截断
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    写入文件内容的封装。

    会自动创建父目录，确保路径存在。
    返回写入的字节数作为确认。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件内容的封装，精确替换一段文本。

    使用字符串替换（只替换第一次出现），如果找不到目标文本则报错。
    这是一个简单但安全的编辑方式。
    """
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text not found in {path}"
        # count=1 表示只替换第一次出现
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 定义可用的工具列表，遵循 OpenAI 工具调用格式
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a Windows cmd-compatible shell command (blocking).",
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
                    "limit": {"type": "integer"}},
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
                    "content": {"type": "string"}},
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
                    "new_text": {"type": "string"}},
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "background_run",
            "description": "Run command in a background thread. Returns task_id immediately.",
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
            "name": "check_background",
            "description": "Check background task status. Omit task_id to list all.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
            },
        },
    },
]


# 工具处理函数映射表，根据工具名称分派到对应的处理函数
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "background_run": lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}


def build_assistant_message(message) -> dict:
    """
    将 API 返回的 assistant 消息对象转换为标准的字典格式。
    处理普通文本消息和工具调用两种情况。
    """
    assistant_message = {
        "role": "assistant",
        "content": message.content or "",  # 如果没有内容则为空字符串
    }

    # 如果消息中包含工具调用，则添加 tool_calls 字段
    tool_calls = message.tool_calls or []
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call.id,  # 工具调用的唯一 ID
                "type": "function",
                "function": {
                    "name": tool_call.function.name,  # 调用的函数名
                    "arguments": tool_call.function.arguments,  # 函数参数字符串
                },
            }
            for tool_call in tool_calls
        ]

    return assistant_message


def dispatch_tool(tool_name: str, arguments: dict) -> str:
    """
    分派工具调用到对应的处理函数。

    从 TOOL_HANDLERS 字典中查找处理器，
    捕获异常并返回格式化的错误信息。
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"

    try:
        return str(handler(**arguments))
    except Exception as e:
        return f"Error: {e}"


def agent_loop(messages: list[dict]) -> None:
    """
    主 Agent 循环，处理单轮对话。

    与之前版本的关键区别：
    每次 LLM 调用前，会先从后台管理器取出已完成任务的通知，
    并作为消息注入对话，让模型知道后台任务已完成。

    流程：
    1. 检查并注入后台任务完成通知
    2. 调用 LLM 生成回复
    3. 处理工具调用
    4. 收集工具结果并追加到消息历史
    5. 重复直到没有工具调用
    """
    while True:
        # 关键步骤：检查后台任务完成通知
        notifs = BG.drain_notifications()
        if notifs and messages:
            # 将通知格式化为消息注入对话
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            messages.append(
                {
                    "role": "user",
                    "content": f"<background-results>\n{notif_text}\n</background-results>",
                }
            )

        # 构建请求消息，包含系统提示和完整对话历史
        request_messages = [{"role": "system", "content": SYSTEM}, *messages]

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=TOOLS,
            tool_choice="auto",  # 让模型自动决定是否调用工具
        )
        message = response.choices[0].message

        # 将 assistant 消息追加到历史
        messages.append(build_assistant_message(message))

        # 如果有文本内容，打印出来
        if message.content:
            log(message.content)

        # 获取工具调用列表
        tool_calls = message.tool_calls or []
        # 如果没有工具调用，说明对话结束（模型直接回复了文本）
        if not tool_calls:
            return

        # 用于存储所有工具结果
        results = []

        # 遍历处理每个工具调用
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                # 解析工具参数（JSON 字符串）
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                # 分派到对应的处理函数
                output = dispatch_tool(tool_name, arguments)

            # 打印工具调用日志
            log(f"> {tool_name}:")
            log(str(output)[:200])

            # 收集工具结果
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": str(output),
                }
            )

        # 将所有工具结果追加到消息历史
        messages.extend(results)


if __name__ == "__main__":
    # 初始化空的消息历史列表
    history: list[dict] = []

    # 主循环：持续读取用户输入
    while True:
        try:
            # 打印提示符，等待用户输入（带颜色的提示符）
            query = input("\033[36mrepro-s08 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C 或 Ctrl+D 退出
            break

        # q、exit 或空行都退出
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 将用户消息追加到历史
        history.append({"role": "user", "content": query})

        # 执行 agent 循环处理这轮对话
        agent_loop(history)
        log("")
