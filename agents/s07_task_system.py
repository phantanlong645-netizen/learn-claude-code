#!/usr/bin/env python3
# Harness: persistent tasks -- goals that outlive any single conversation.
"""
s07_task_system.py - 任务系统

任务以 JSON 文件形式持久化在 .tasks/ 目录中，因此可以在上下文压缩中存活。
每个任务都有依赖图（blockedBy 字段）。

    .tasks/
      task_1.json  {"id":1, "subject":"...", "status":"completed", ...}
      task_2.json  {"id":2, "blockedBy":[1], "status":"pending", ...}
      task_3.json  {"id":3, "blockedBy":[2], ...}

    依赖解析示意:
    +----------+     +----------+     +----------+
    | task 1   | --> | task 2   | --> | task 3   |
    | complete |     | blocked  |     | blocked  |
    +----------+     +----------+     +----------+
         |                ^
         +--- 完成 task 1 后会自动从 task 2 的 blockedBy 中移除

核心思想："状态存活在对话之外 —— 因为它存储在磁盘上，而非对话中。"
"""

import json
import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv(override=True)

# 如果设置了 ANTHROPIC_BASE_URL，则移除 ANTHROPIC_AUTH_TOKEN（避免冲突）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()  # 工作目录，作为安全路径的根目录

# 创建 Anthropic 客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]  # 从环境变量获取模型 ID

# 任务文件的存储目录
TASKS_DIR = WORKDIR / ".tasks"

# 系统提示词
SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


# -- TaskManager: 任务 CRUD 操作，带依赖图，持久化为 JSON 文件 --
class TaskManager:
    """
    任务管理器：负责创建、读取、更新、删除任务，
    并维护任务之间的依赖关系。
    """

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)  # 确保目录存在
        # 计算下一个可用的任务 ID
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """
        获取当前最大的任务 ID。
        通过扫描 .tasks/ 目录下所有 task_*.json 文件来计算。
        """
        # 遍历所有 task_*.json 文件，提取 ID
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0  # 如果没有任务，返回 0

    def _load(self, task_id: int) -> dict:
        """
        从磁盘加载指定 ID 的任务。
        """
        path = self.dir / f"task_{task_id}.json"
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")
        return json.loads(path.read_text())

    def _save(self, task: dict):
        """
        将任务保存到磁盘（JSON 文件）。
        """
        path = self.dir / f"task_{task['id']}.json"
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False))

    def create(self, subject: str, description: str = "") -> str:
        """
        创建新任务。

        Args:
            subject: 任务主题/标题
            description: 任务详细描述（可选）

        Returns:
            返回创建的任务的 JSON 字符串
        """
        task = {
            "id": self._next_id,          # 分配新 ID
            "subject": subject,            # 任务主题
            "description": description,    # 详细描述
            "status": "pending",           # 默认状态为 pending
            "blockedBy": [],               # 默认没有被阻塞的任务
            "owner": "",                   # 所有者（暂未使用）
        }
        self._save(task)                   # 保存到磁盘
        self._next_id += 1                 # 下一个任务的 ID +1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        """
        获取指定任务 ID 的详细信息。

        Returns:
            任务详情的 JSON 字符串
        """
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(self, task_id: int, status: str = None,
               add_blocked_by: list = None, remove_blocked_by: list = None) -> str:
        """
        更新任务的状态或依赖关系。

        Args:
            task_id: 任务 ID
            status: 新状态（pending/in_progress/completed）
            add_blocked_by: 添加阻塞此任务的任务 ID 列表
            remove_blocked_by: 从阻塞列表中移除的任务 ID 列表

        Returns:
            更新后任务的 JSON 字符串
        """
        task = self._load(task_id)

        # 更新状态
        if status:
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            # 如果任务标记为完成，清除所有依赖
            if status == "completed":
                self._clear_dependency(task_id)

        # 添加阻塞任务
        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))

        # 移除阻塞任务
        if remove_blocked_by:
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]

        self._save(task)
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int):
        """
        清除已完成任务的依赖关系。

        当一个任务完成时，需要从所有其他任务的 blockedBy 列表中移除它。
        这是一个"后向清理"操作。

        Args:
            completed_id: 已完成的任务 ID
        """
        for f in self.dir.glob("task_*.json"):
            task = json.loads(f.read_text())
            if completed_id in task.get("blockedBy", []):
                task["blockedBy"].remove(completed_id)
                self._save(task)

    def list_all(self) -> str:
        """
        列出所有任务。

        Returns:
            格式化后的任务列表字符串，包含状态标记和依赖信息
        """
        tasks = []
        # 按 ID 排序遍历所有任务文件
        files = sorted(
            self.dir.glob("task_*.json"),
            key=lambda f: int(f.stem.split("_")[1])
        )
        for f in files:
            tasks.append(json.loads(f.read_text()))

        if not tasks:
            return "No tasks."

        lines = []
        for t in tasks:
            # 根据状态选择标记符号
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}.get(t["status"], "[?]")
            # 如果有阻塞任务，显示依赖信息
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{marker} #{t['id']}: {t['subject']}{blocked}")

        return "\n".join(lines)


# 全局任务管理器实例
TASKS = TaskManager(TASKS_DIR)


# -- 基础工具实现 --

def safe_path(p: str) -> Path:
    """
    安全路径解析，防止通过路径遍历访问工作目录之外的文件。
    """
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """
    执行 shell 命令。
    包含危险命令黑名单检查，防止执行危险操作。
    """
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
    """
    读取文件内容。
    支持行数限制，超过限制的部分会用省略号标注。
    """
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """
    写入文件内容。
    会自动创建父目录。
    """
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """
    编辑文件内容，精确替换一段文本。
    只替换第一次出现。
    """
    try:
        fp = safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具处理函数映射表
TOOL_HANDLERS = {
    "bash":        lambda **kw: run_bash(kw["command"]),
    "read_file":   lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file":  lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":   lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 任务相关工具
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status"), kw.get("addBlockedBy"), kw.get("removeBlockedBy")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}


# 工具定义列表（用于发送给模型）
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "task_create", "description": "Create a new task.",
     "input_schema": {"type": "object", "properties": {"subject": {"type": "string"}, "description": {"type": "string"}}, "required": ["subject"]}},
    {"name": "task_update", "description": "Update a task's status or dependencies.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}, "addBlockedBy": {"type": "array", "items": {"type": "integer"}}, "removeBlockedBy": {"type": "array", "items": {"type": "integer"}}}, "required": ["task_id"]}},
    {"name": "task_list", "description": "List all tasks with status summary.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "task_get", "description": "Get full details of a task by ID.",
     "input_schema": {"type": "object", "properties": {"task_id": {"type": "integer"}}, "required": ["task_id"]}},
]


def agent_loop(messages: list):
    """
    主 Agent 循环，处理用户请求。

    流程:
    1. 调用 LLM 获取响应
    2. 如果有工具调用，执行工具
    3. 将工具结果返回给 LLM
    4. 重复直到 LLM 不再调用工具
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 将助手消息添加到历史
        messages.append({"role": "assistant", "content": response.content})

        # 如果不是工具调用，说明对话结束
        if response.stop_reason != "tool_use":
            return

        # 处理工具调用
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

        # 将工具结果作为新消息添加（触发下一轮 LLM 调用）
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []  # 对话历史

    while True:
        try:
            query = input("\033[36ms07 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # q、exit 或空行退出
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 将用户消息添加到历史
        history.append({"role": "user", "content": query})

        # 执行 agent 循环
        agent_loop(history)

        # 打印助手的文本回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
