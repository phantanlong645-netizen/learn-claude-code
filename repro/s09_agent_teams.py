#!/usr/bin/env python3
"""
Repro for s09: Agent Teams

多智能体团队协作演示，遵循原始 s09 课程内容：
1. 持久化具名 Agent（Teammate）
2. 基于文件的 JSONL 邮箱实现进程间通信
3. 每个 teammate 运行在独立线程中

唯一针对特定 provider 的改动是 OpenAI 兼容的工具调用格式。
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import threading
import time
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

# 团队目录和收件箱目录
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# Lead 的系统提示词
SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

# 支持的消息类型集合
VALID_MSG_TYPES = {
    "message",                  # 普通私信
    "broadcast",                # 广播给所有人
    "shutdown_request",         # 请求关闭（s10）
    "shutdown_response",        # 关闭响应（s10）
    "plan_approval_response",   # 计划审批（s10）
}


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


# -- MessageBus: 每个 teammate 一个 JSONL 收件箱 --
class MessageBus:
    """
    消息总线，管理基于文件的邮箱系统。

    每个 teammate 有一个对应的 .jsonl 文件作为收件箱。
    消息以 JSONL 格式追加（append-only），读取时清空。

    优点：
    - 简单可靠，不依赖外部消息队列
    - 可以跨进程共享（如需要）
    - 有持久化，程序重启后消息不丢失
    """

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """
        发送消息给某个 teammate。

        Args:
            sender: 发送者名字
            to: 接收者名字
            content: 消息内容
            msg_type: 消息类型（默认 message）
            extra: 额外字段

        Returns:
            确认消息字符串
        """
        # 验证消息类型是否合法
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        # 构建消息对象
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),  # 时间戳
        }
        if extra:
            msg.update(extra)

        # 追加写入到接收者的收件箱文件
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:  # "a" 模式追加
            f.write(json.dumps(msg) + "\n")

        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空某个收件箱。

        读完后文件被清空（drain），避免重复处理。

        Args:
            name: 收件箱名字（如 "alice", "lead"）

        Returns:
            消息列表
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []

        # 读取所有行并解析为 JSON 对象
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))

        # 清空收件箱（drain）
        inbox_path.write_text("")

        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        """
        广播消息给所有 teammate（除了发送者自己）。

        Args:
            sender: 发送者名字
            content: 消息内容
            teammates: 所有 teammate 名字列表

        Returns:
            确认消息
        """
        count = 0
        for name in teammates:
            if name != sender:  # 不发给自己
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# 全局消息总线实例
BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: 持久化具名 Agent 管理 --
class TeammateManager:
    """
    团队管理器。

    核心功能：
    1. 维护团队配置（config.json）
    2. Spawn/shutdown teammate
    3. 追踪 teammate 状态（working/idle/shutdown）

    与 s04 Subagent 的区别：
    - Teammate 是持久化的，可以多次处理任务
    - 有状态管理（idle 可以重新激活）
    - 通过邮箱通信而非直接返回
    """

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()   # 加载配置
        self.threads = {}                   # 线程字典：name -> Thread

    def _load_config(self) -> dict:
        """加载团队配置，如果不存在则返回默认配置。"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """保存团队配置到文件。"""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """根据名字查找团队成员。"""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        启动一个 teammate（如果已存在且空闲，则重新激活）。

        Args:
            name: teammate 名字
            role: 角色（如 "coder", "tester"）
            prompt: 初始任务描述

        Returns:
            确认消息
        """
        member = self._find_member(name)

        if member:
            # 成员已存在
            if member["status"] not in ("idle", "shutdown"):
                # 如果正在工作中，不能重复启动
                return f"Error: '{name}' is currently {member['status']}"
            # 重新激活
            member["status"] = "working"
            member["role"] = role
        else:
            # 新成员
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)

        self._save_config()

        # 创建并启动线程
        thread = threading.Thread(
            target=self._teammate_loop,      # 线程执行函数
            args=(name, role, prompt),        # 传递给函数的参数
            daemon=True,                      # 守护线程
        )
        self.threads[name] = thread
        thread.start()

        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        Teammate 的主循环，运行在独立线程中。

        流程：
        1. 读收件箱
        2. 调用 LLM
        3. 处理工具调用
        4. 重复直到完成或达到迭代上限

        Args:
            name: teammate 名字
            role: 角色
            prompt: 初始任务
        """
        # 构建系统提示词
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )

        # 初始化消息历史，包含初始任务
        messages = [{"role": "user", "content": prompt}]

        # 获取 teammate 可用的工具
        tools = self._teammate_tools()

        # 最多迭代 50 次（防止无限循环）
        for _ in range(50):
            # 1. 读收件箱
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                # 将消息转换为对话格式追加
                messages.append({"role": "user", "content": json.dumps(msg)})

            # 2. 调用 LLM（OpenAI 兼容格式）
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception:
                # API 出错，退出循环
                break

            # 获取回复消息
            message = response.choices[0].message

            # 添加助手回复
            messages.append({"role": "assistant", "content": message.content or ""})

            # 处理工具调用
            if message.tool_calls:
                results = []
                for tool_call in message.tool_calls:
                    # 执行工具
                    output = self._exec(name, tool_call.function.name,
                                       json.loads(tool_call.function.arguments or "{}"))
                    log(f"  [{name}] {tool_call.function.name}: {str(output)[:120]}")
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(output),
                    })
                # 将工具结果追加到消息历史
                messages.extend(results)
            else:
                # 如果不是工具调用，说明任务完成
                break

        # 循环结束，标记为空闲
        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行 teammate 的工具调用。

        包括基础工具（bash, read_file 等）和通信工具（send_message, read_inbox）。

        Args:
            sender: 调用者的名字（用于 send_message）
            tool_name: 工具名
            args: 工具参数

        Returns:
            工具执行结果
        """
        # 基础工具（与 s02 相同）
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])

        # 通信工具
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        返回 teammate 可用的工具列表（OpenAI 格式）。

        包括：
        - 基础工具：bash, read_file, write_file, edit_file
        - 通信工具：send_message, read_inbox
        """
        return [
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
                    "name": "send_message",
                    "description": "Send message to a teammate.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "content": {"type": "string"},
                            "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)},
                        },
                        "required": ["to", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_inbox",
                    "description": "Read and drain your inbox.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def list_all(self) -> str:
        """列出所有 teammates 的状态。"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """返回所有成员名字列表。"""
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器实例
TEAM = TeammateManager(TEAM_DIR)


# -- 基础工具实现（与 s02 相同）--

def _safe_path(p: str) -> Path:
    """安全路径解析，防止路径遍历。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行 shell 命令（阻塞式）。"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    """读取文件内容。"""
    try:
        lines = _safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """写入文件内容。"""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件内容。"""
    try:
        fp = _safe_path(path)
        c = fp.read_text(encoding="utf-8")
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead 工具分派（9个工具，OpenAI 格式）--

TOOL_HANDLERS = {
    # 基础工具
    "bash":            lambda **kw: _run_bash(kw["command"]),
    "read_file":       lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":      lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":       lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 团队管理工具
    "spawn_teammate":  lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":  lambda **kw: TEAM.list_all(),

    # 通信工具
    "send_message":    lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":      lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":       lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
}

# Lead 可用的工具列表（OpenAI 格式）
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
            "name": "spawn_teammate",
            "description": "Spawn a persistent teammate that runs in its own thread.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["name", "role", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List all teammates with name, role, status.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a teammate's inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)},
                },
                "required": ["to", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_inbox",
            "description": "Read and drain the lead's inbox.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast",
            "description": "Send a message to all teammates.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
]


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


def agent_loop(messages: list):
    """
    Lead 的主 Agent 循环。

    与 s08/s09 的关键区别：
    - 每次循环开始时检查 Lead 的收件箱
    - 如果有消息，注入到对话中让 LLM 处理
    - 支持 spawn_teammate 等团队管理工具
    """
    while True:
        # 检查 Lead 的收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:
            # 将消息注入到对话中
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })

        # 调用 LLM（OpenAI 兼容格式）
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",
        )

        # 获取回复消息
        message = response.choices[0].message

        # 将助手回复追加到历史
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
    history = []

    while True:
        try:
            query = input("\033[36mrepro-s09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 特殊命令
        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            # 查看团队状态
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            # 查看收件箱
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        # 正常对话
        history.append({"role": "user", "content": query})
        agent_loop(history)
        log("")
