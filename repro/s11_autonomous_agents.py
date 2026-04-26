#!/usr/bin/env python3
"""
Repro for s11: Autonomous Agents

自主智能体演示，构建在 s10 基础上：
1. 空闲轮询机制 - Teammate 空闲时定期检查任务
2. 任务看板 - .tasks/ 目录存储任务
3. 自动领取 - Teammate 可以自动认领任务
4. 身份重注入 - 上下文压缩后恢复身份信息

s10: spawn -> work -> idle（等待消息，不会主动找事）
s11: spawn -> work -> idle -> poll tasks -> 自动领取 -> work

Teammate 生命周期（s11）：
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use 或 调用 idle 工具
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown

核心概念：
1. idle 工具 - Teammate 主动调用表示"我没活了，进入空闲"
2. poll 机制 - 空闲时定期检查收件箱和任务看板
3. auto-claim - 自动从任务看板领取任务
4. identity re-injection - 上下文压缩后重新注入身份信息

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

# 加载 .env 环境变量
load_dotenv(override=True)

# 为 Windows 控制台设置 UTF-8 编码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 工作目录
WORKDIR = Path.cwd()

# OpenAI 兼容客户端
client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)

# 使用的模型
MODEL = os.getenv("MODEL_ID", "MiniMax-M2.2")

# 目录配置
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"

# 轮询配置
POLL_INTERVAL = 5   # 空闲时每隔多少秒检查一次
IDLE_TIMEOUT = 60   # 空闲超时时间（秒），超时后自动 shutdown

# Lead 系统提示词
SYSTEM = f"""You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves.

当没有具体任务时，可以让 teammates 去任务看板认领任务。"""

# 支持的消息类型
VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- 共享状态 --
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


def log(*parts) -> None:
    """打印带位置信息的日志"""
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    location = f"[{Path(__file__).name}:{caller.f_lineno}]" if caller else f"[{Path(__file__).name}]"
    text = " ".join(str(part) for part in parts)
    print(f"{location} {text}")


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    """消息总线，每个 teammate 一个 .jsonl 收件箱"""

    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
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
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- Task Board: 任务看板相关函数 --

def scan_unclaimed_tasks() -> list:
    """
    扫描任务看板，返回所有未认领的任务。

    未认领条件：
    - status == "pending"
    - 没有 owner
    - 没有 blockedBy（没有被其他任务阻塞）

    Returns:
        未认领任务列表
    """
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        task = json.loads(f.read_text())
        if (task.get("status") == "pending"
                and not task.get("owner")
                and not task.get("blockedBy")):
            unclaimed.append(task)
    return unclaimed


def claim_task(task_id: int, owner: str) -> str:
    """
    认领任务。

    使用锁保证并发安全，防止多个 teammate 同时认领同一个任务。

    Args:
        task_id: 任务 ID
        owner: 认领者名字

    Returns:
        成功或错误信息
    """
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        task = json.loads(path.read_text())
        if task.get("owner"):
            existing_owner = task.get("owner") or "someone else"
            return f"Error: Task {task_id} has already been claimed by {existing_owner}"
        if task.get("status") != "pending":
            status = task.get("status")
            return f"Error: Task {task_id} cannot be claimed because its status is '{status}'"
        if task.get("blockedBy"):
            return f"Error: Task {task_id} is blocked by other task(s) and cannot be claimed yet"
        task["owner"] = owner
        task["status"] = "in_progress"
        path.write_text(json.dumps(task, indent=2))
    return f"Claimed task #{task_id} for {owner}"


def make_identity_block(name: str, role: str, team_name: str) -> dict:
    """
    创建身份信息块，用于上下文压缩后重注入身份。

    当消息历史被压缩后，需要重新告诉 LLM "你是谁"。

    Args:
        name: 名字
        role: 角色
        team_name: 团队名

    Returns:
        用户消息字典
    """
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


# -- TeammateManager: 自主智能体管理 --
class TeammateManager:
    """
    自主智能体管理器，构建在 s10 基础上：

    新增功能：
    1. idle 工具 - Teammate 主动进入空闲状态
    2. 空闲轮询 - 定期检查收件箱和任务看板
    3. auto-claim - 自动领取任务
    4. 超时 shutdown - 空闲超时后自动关闭
    """

    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def _set_status(self, name: str, status: str):
        """更新成员状态"""
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        """
        启动一个自主 Teammate。

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
            target=self._loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _loop(self, name: str, role: str, prompt: str):
        """
        Teammate 的主循环（包含 WORK + IDLE 两个阶段）。

        流程：
        1. WORK 阶段：执行任务，调用 LLM
        2. IDLE 阶段：轮询收件箱和任务看板
           - 有消息 -> 继续 WORK
           - 有未认领任务 -> auto-claim -> 继续 WORK
           - 超时 -> shutdown

        Args:
            name: teammate 名字
            role: 角色
            prompt: 初始任务
        """
        team_name = self.config["team_name"]

        # 系统提示词（s11 新增 idle 工具说明）
        sys_prompt = (
            f"You are '{name}', role: {role}, team: {team_name}, at {WORKDIR}. "
            f"Use idle tool when you have no more work. You will auto-claim new tasks."
        )

        messages = [{"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # ========== WORK 阶段 ==========
            for _ in range(50):
                # 1. 检查收件箱
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})

                # 2. 调用 LLM
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": sys_prompt}, *messages],
                        tools=tools,
                        tool_choice="auto",
                    )
                except Exception:
                    self._set_status(name, "idle")
                    return

                message = response.choices[0].message
                messages.append({"role": "assistant", "content": message.content or ""})

                # 3. 如果不是工具调用，退出循环进入 IDLE
                if not message.tool_calls:
                    break

                # 4. 处理工具调用
                results = []
                idle_requested = False
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")

                    if tool_name == "idle":
                        idle_requested = True
                        output = "Entering idle phase. Will poll for new tasks."
                    else:
                        output = self._exec(name, tool_name, args)

                    log(f"  [{name}] {tool_name}: {str(output)[:120]}")
                    results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(output),
                    })

                messages.append({"role": "user", "content": results})

                # 如果调用了 idle 工具，立即进入 IDLE 阶段
                if idle_requested:
                    break

            # ========== IDLE 阶段 ==========
            self._set_status(name, "idle")
            resume = False

            # 最多轮询 N 次（总共 IDLE_TIMEOUT 秒）
            polls = IDLE_TIMEOUT // max(POLL_INTERVAL, 1)
            for _ in range(polls):
                time.sleep(POLL_INTERVAL)

                # 1. 检查收件箱
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break

                # 2. 扫描任务看板，寻找未认领任务
                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    result = claim_task(task["id"], name)
                    if result.startswith("Error:"):
                        continue

                    # 成功认领，注入任务信息
                    task_prompt = (
                        f"<auto-claimed>Task #{task['id']}: {task['subject']}\n"
                        f"{task.get('description', '')}</auto-claimed>"
                    )

                    # 如果消息历史太短（被压缩过），重新注入身份信息
                    if len(messages) <= 3:
                        messages.insert(0, make_identity_block(name, role, team_name))
                        messages.insert(1, {"role": "assistant", "content": f"I am {name}. Continuing."})

                    messages.append({"role": "user", "content": task_prompt})
                    messages.append({"role": "assistant", "content": f"Claimed task #{task['id']}. Working on it."})
                    resume = True
                    break

            # 3. 如果没有恢复工作的理由，超时 shutdown
            if not resume:
                self._set_status(name, "shutdown")
                return

            self._set_status(name, "working")

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        """
        执行工具调用。

        新增工具：
        - idle: 进入空闲状态
        - claim_task: 领取任务
        """
        # 基础工具
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

        # 协议工具（与 s10 相同）
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if args["approve"] else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": args["approve"]},
            )
            return f"Shutdown {'approved' if args['approve'] else 'rejected'}"

        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for approval."

        # -- s11 新增工具 --

        # idle: 主动进入空闲状态
        if tool_name == "idle":
            return "Entering idle phase."

        # claim_task: 从任务看板领取任务
        if tool_name == "claim_task":
            return claim_task(args["task_id"], sender)

        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        """
        Teammate 可用的工具列表（OpenAI 格式）。

        新增：
        - idle: 进入空闲状态
        - claim_task: 领取任务
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
            # 协议工具
            {
                "type": "function",
                "function": {
                    "name": "shutdown_response",
                    "description": "Respond to a shutdown request.",
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
                    "description": "Submit a plan for lead approval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan": {"type": "string"}},
                        "required": ["plan"],
                    },
                },
            },
            # -- s11 新增工具 --
            {
                "type": "function",
                "function": {
                    "name": "idle",
                    "description": "Signal that you have no more work. Enters idle polling phase.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "claim_task",
                    "description": "Claim a task from the task board by ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "integer"}},
                        "required": ["task_id"],
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


# 全局团队管理器
TEAM = TeammateManager(TEAM_DIR)


# -- 基础工具实现 --

def _safe_path(p: str) -> Path:
    """安全路径解析"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    """执行 shell 命令"""
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=120,
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


# -- Lead 协议处理函数 --

def handle_shutdown_request(teammate: str) -> str:
    """发送关闭请求"""
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}'"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    """审批计划"""
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
    """查询关闭请求状态"""
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# -- Lead 工具分派（14 个工具）--

TOOL_HANDLERS = {
    # 基础工具
    "bash":              lambda **kw: _run_bash(kw["command"]),
    "read_file":         lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file":        lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file":         lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),

    # 团队管理
    "spawn_teammate":    lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates":    lambda **kw: TEAM.list_all(),

    # 通信工具
    "send_message":      lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox":        lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast":         lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),

    # 协议工具
    "shutdown_request":  lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval":     lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),

    # s11 新增
    "idle":              lambda **kw: "Lead does not idle.",
    "claim_task":        lambda **kw: claim_task(kw["task_id"], "lead"),
}

# Lead 工具列表（OpenAI 格式，14 个工具）
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
    # 团队管理（2个）
    {
        "type": "function",
        "function": {
            "name": "spawn_teammate",
            "description": "Spawn an autonomous teammate.",
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
    # 协议工具（3个）
    {
        "type": "function",
        "function": {
            "name": "shutdown_request",
            "description": "Request a teammate to shut down.",
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
            "description": "Check shutdown request status.",
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
            "description": "Approve or reject a teammate's plan.",
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
    # s11 新增（2个）
    {
        "type": "function",
        "function": {
            "name": "idle",
            "description": "Enter idle state (for lead -- rarely used).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claim_task",
            "description": "Claim a task from the board by ID.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
    },
]


def build_assistant_message(message) -> dict:
    """转换 assistant 消息格式"""
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
    """分派工具调用"""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"
    try:
        return str(handler(**arguments))
    except Exception as e:
        return f"Error: {e}"


def list_tasks() -> str:
    """
    列出所有任务（用于 /tasks 命令）。
    """
    TASKS_DIR.mkdir(exist_ok=True)
    tasks = []
    for f in sorted(TASKS_DIR.glob("task_*.json")):
        t = json.loads(f.read_text())
        marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
        owner = f" @{t['owner']}" if t.get("owner") else ""
        tasks.append(f"  {marker} #{t['id']}: {t['subject']}{owner}")
    if not tasks:
        return "No tasks."
    return "\n".join(tasks)


def agent_loop(messages: list):
    """Lead 的主 Agent 循环"""
    while True:
        # 检查收件箱
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

        # 打印文本
        if message.content:
            log(message.content)

        # 处理工具调用
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
            query = input("\033[36mrepro-s11 >> \033[0m")
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
        if query.strip() == "/tasks":
            print(list_tasks())
            continue

        # 正常对话
        history.append({"role": "user", "content": query})
        agent_loop(history)
        log("")
