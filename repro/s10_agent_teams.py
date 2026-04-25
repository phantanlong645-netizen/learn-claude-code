#!/usr/bin/env python3
"""
Repro for s10: Team Protocols

多智能体团队协议演示，构建在 s09 基础上：
1. Shutdown 协议 - 优雅关闭 teammate
2. Plan Approval 协议 - teammate 提交计划，lead 审批
3. 使用 request_id 关联请求和响应

s09: spawn -> work -> idle -> work -> ... -> shutdown (手动)
s10: 新增 shutdown_request/shutdown_response 协议流程
     新增 plan_approval 计划审批流程

通信流程：
    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request    |          |                     |
    | {request_id: abc}   | -------> | 接收请求，决定是否   |
    +---------------------+          | 批准或拒绝           |
                                     +---------------------+
    +---------------------+          +---------------------+
    | shutdown_response   | <------- | shutdown_response   |
    | {request_id: abc,   |          | {request_id: abc,   |
    |  approve: true}     |          |  approve: true}     |
    +---------------------+          +---------------------+
            │
            v
    状态改为 "shutdown"，线程退出

    Plan Approval 流程：

    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval       |          |                     |
    | submit: {plan:...} | -------> | 审查计划文本        |
    +---------------------+          | 批准或拒绝           |
                                     +---------------------+
    +---------------------+          +---------------------+
    | plan_approval_resp  | <------- | plan_approval_resp  |
    | {approve: true}     |          | {request_id,        |
    +---------------------+          |  approve: true}     |
                                     +---------------------+

核心概念：request_id 关联模式
- 每条请求生成唯一 request_id
- 响应中携带相同 request_id 用于追踪
- 使用 threading.Lock 保证线程安全

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
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

# 使用的模型 ID
MODEL = os.getenv("MODEL_ID", "MiniMax-M2.2")

# 团队目录和收件箱目录
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

# Lead 的系统提示词（s10 新增协议相关指令）
SYSTEM = f"""You are a team lead at {WORKDIR}.
Manage teammates with shutdown and plan approval protocols.

协议说明：
1. Shutdown 协议：当需要关闭 teammate 时，使用 shutdown_request 工具
2. Plan Approval 协议：teammate 会提交计划，你需要用 plan_approval 工具审批"""

# 支持的消息类型集合（s10 新增 shutdown 和 plan 相关类型）
VALID_MSG_TYPES = {
    "message",                  # 普通私信
    "broadcast",               # 广播给所有人
    "shutdown_request",         # 关闭请求
    "shutdown_response",        # 关闭响应
    "plan_approval_response",   # 计划审批响应
}

# -- Request trackers: 通过 request_id 关联请求和响应 --
# shutdown_requests 存储所有发出的关闭请求
# plan_requests 存储所有提交的计划审批请求
shutdown_requests = {}
plan_requests = {}

# 线程锁，保证多线程访问共享字典时的线程安全
_tracker_lock = threading.Lock()


def log(*parts) -> None:
    """
    打印带位置信息的日志，方便调试和追踪代码执行路径。
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    location = f"[{Path(__file__).name}:{caller.f_lineno}]" if caller else f"[{Path(__file__).name}]"
    text = " ".join(str(part) for part in parts)
    print(f"{location} {text}")


# -- MessageBus: 每个 teammate 一个 JSONL 收件箱 --
class MessageBus:
    """
    消息总线，管理基于文件的邮箱系统。

    与 s09 完全相同，每个 teammate 有一个 .jsonl 文件作为收件箱。
    """

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        """
        发送消息给某个 teammate。

        Args:
            sender: 发送者名字
            to: 接收者名字
            content: 消息内容
            msg_type: 消息类型（默认 message）
            extra: 额外字段（如 request_id）

        Returns:
            确认消息字符串
        """
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"

        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)

        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        """
        读取并清空某个收件箱（drain 模式）。

        Args:
            name: 收件箱名字

        Returns:
            消息列表
        """
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []

        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))

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
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


# 全局消息总线实例
BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: 持久化具名 Agent 管理 + 协议支持 --
class TeammateManager:
    """
    团队管理器，构建在 s09 基础上：

    新增功能：
    1. shutdown_response 工具 - teammate 响应关闭请求
    2. plan_approval 工具 - teammate 提交计划待审批
    3. should_exit 标志 - 控制线程退出

    与 s09 的区别：
    - Teammate 可以响应 shutdown_request 并优雅退出
    - Teammate 可以提交计划给 lead 审批
    """

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        """加载团队配置"""
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        """保存团队配置到文件"""
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        """根据名字查找团队成员"""
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        启动一个 teammate（如果已存在且空闲或关闭，则重新激活）。

        Args:
            name: teammate 名字
            role: 角色
            prompt: 初始任务描述

        Returns:
            确认消息
        """
        member = self._find_member(name)

        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)

        self._save_config()

        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()

        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        """
        Teammate 的主循环，运行在独立线程中。

        新增：shutdown_response 和 plan_approval 协议处理

        Args:
            name: teammate 名字
            role: 角色
            prompt: 初始任务
        """
        # 系统提示词（s10 新增协议指令）
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"After completing tasks, use send_message to report results to lead. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )

        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        # should_exit 标志：收到 shutdown_response 且 approve=True 时设为 True
        should_exit = False

        for _ in range(50):
            # 1. 读收件箱
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})

            # 2. 检查是否需要退出
            if should_exit:
                break

            # 3. 调用 LLM
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": sys_prompt}, *messages],
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception:
                break

            message = response.choices[0].message
            messages.append({"role": "assistant", "content": message.content or ""})

            # 4. 处理工具调用
            if message.tool_calls:
                results = []
                for tool_call in message.tool_calls:
                    output = self._exec(name, tool_call.function.name,
                                       json.loads(tool_call.function.arguments or "{}"))
                    log(f"  [{name}] {tool_call.function.name}: {str(output)[:120]}")

                    # 检查是否是批准关闭的响应
                    if tool_call.function.name == "shutdown_response":
                        args = json.loads(tool_call.function.arguments or "{}")
                        if args.get("approve"):
                            should_exit = True

                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": str(output),
                    })
                messages.extend(results)
            else:
                # 没有工具调用，说明任务完成或对话结束
                break

        # 5. 循环结束，更新状态
        member = self._find_member(name)
        if member:
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行 teammate 的工具调用。

        新增工具：
        - shutdown_response: 响应关闭请求
        - plan_approval: 提交计划审批

        Args:
            sender: 调用者名字
            tool_name: 工具名
            args: 工具参数

        Returns:
            工具执行结果
        """
        # 基础工具（与 s02/s09 相同）
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"], args.get("limit"))
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])

        # 通信工具（与 s09 相同）
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)

        # -- s10 新增协议工具 --

        # shutdown_response: 响应关闭请求
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            # 更新 shutdown_requests 状态
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            # 发消息给 lead 告知响应
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"

        # plan_approval: 提交计划审批
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]  # 生成唯一请求 ID
            # 记录计划请求
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            # 发消息给 lead 审批
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        返回 teammate 可用的工具列表（OpenAI 格式）。

        新增：shutdown_response, plan_approval
        """
        return [
            # 基础工具
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
            # 通信工具
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
                            "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}},
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
            # -- s10 新增协议工具 --
            {
                "type": "function",
                "function": {
                    "name": "shutdown_response",
                    "description": "Respond to a shutdown request. Approve to shut down, reject to keep working.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {"type": "string"},
                            "approve": {"type": "boolean"},
                            "reason": {"type": "string"}},
                        "required": ["request_id", "approve"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_approval",
                    "description": "Submit a plan for lead approval. Provide plan text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {"type": "string"}},
                        "required": ["plan"],
                    },
                },
            },
        ]

    def list_all(self) -> str:
        """列出所有 teammates 的状态"""
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        """返回所有成员名字列表"""
        return [m["name"] for m in self.config["members"]]


# 全局团队管理器实例
TEAM = TeammateManager(TEAM_DIR)


# -- 基础工具实现（与 s02/s09 相同）--

def _safe_path(p: str) -> Path:
    """安全路径解析，防止路径遍历"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行 shell 命令（阻塞式）"""
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
    """读取文件内容"""
    try:
        lines = _safe_path(path).read_text(encoding="utf-8").splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    """写入文件内容"""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件内容"""
    try:
        fp = _safe_path(path)
        c = fp.read_text(encoding="utf-8")
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead 专用协议处理函数 --

def handle_shutdown_request(teammate: str) -> str:
    """
    处理关闭请求。

    1. 生成唯一 request_id
    2. 记录到 shutdown_requests 追踪器
    3. 发送 shutdown_request 消息给 teammate

    Args:
        teammate: 要关闭的 teammate 名字

    Returns:
        确认消息（包含 request_id）
    """
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """
    处理计划审批。

    1. 根据 request_id 查找对应的计划
    2. 更新计划状态为 approved 或 rejected
    3. 发送 plan_approval_response 消息给提交者

    Args:
        request_id: 计划请求的唯一 ID
        approve: 是否批准
        feedback: 反馈信息

    Returns:
        确认消息
    """
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    """
    查询关闭请求的状态。

    Args:
        request_id: 请求 ID

    Returns:
        请求状态的 JSON 字符串
    """
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# -- Lead 工具分派（12 个工具，OpenAI 格式）--

TOOL_HANDLERS = {
    # 基础工具（与 s09 相同）
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 团队管理工具（与 s09 相同）
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),

    # 通信工具（与 s09 相同）
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":        lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),

    # -- s10 新增协议工具 --
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response":  lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":      lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
}

# Lead 可用的工具列表（OpenAI 格式，12 个工具）
TOOLS = [
    # 基础工具（4个）
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
    # 团队管理工具（2个）
    {
        "type": "function",
        "function": {
            "name": "spawn_teammate",
            "description": "Spawn a persistent teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "prompt": {"type": "string"}},
                "required": ["name", "role", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List all teammates.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # 通信工具（3个）
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a teammate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {"type": "string", "enum": list(VALID_MSG_TYPES)}},
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
    # -- s10 新增协议工具（3个）--
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "Request a teammate to shut down gracefully. Returns a request_id for tracking.",
            "parameters": {
                "type": "object",
                "properties": {"teammate": {"type": "string"}},
                "required": ["teammate"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shutdown_response",
            "description": "Check the status of a shutdown request by request_id.",
            "parameters": {
                "type": "object",
                "properties": {"request_id": {"type": "string"}},
                "required": ["request_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plan_approval",
            "description": "Approve or reject a teammate's plan. Provide request_id + approve + optional feedback.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_id": {"type": "string"},
                    "approve": {"type": "boolean"},
                    "feedback": {"type": "string"}},
                "required": ["request_id", "approve"],
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
    """
    分派工具调用到对应的处理函数。
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

    与 s09 的区别：
    - 支持 shutdown_request 工具
    - 支持 plan_approval 工具
    """
    while True:
        # 检查 Lead 的收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })

        # 调用 LLM
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}, *messages],
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(build_assistant_message(message))

        # 打印文本内容
        if message.content:
            log(message.content)

        # 获取工具调用列表
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return

        # 处理工具调用
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                output = dispatch_tool(tool_name, arguments)

            log(f"> {tool_name}:")
            log(str(output)[:200])

            results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(output),
            })

        messages.extend(results)


if __name__ == "__main__":
    history = []

    while True:
        try:
            query = input("\033[36mrepro-s10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ("q", "exit", ""):
            break
        if query.strip() == "/team":
            print(TEAM.list_all())
            continue
        if query.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2))
            continue

        # 正常对话
        history.append({"role": "user", "content": query})
        agent_loop(history)
        log("")
