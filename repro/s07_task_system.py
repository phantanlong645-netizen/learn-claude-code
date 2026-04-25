#!/usr/bin/env python3
"""
Repro for s07: Task System

任务系统演示，遵循原始 s07 课程内容：
1. 将任务持久化为 .tasks/ 目录下的 JSON 文件
2. 通过 blockedBy 字段追踪任务状态和依赖关系
3. 向模型暴露 CRUD 风格的任务工具

唯一针对特定 provider 的改动是 OpenAI 兼容的工具调用格式。
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

# 加载 .env 环境变量，override=True 表示覆盖系统已有变量
load_dotenv(override=True)

# 为 Windows 控制台设置 UTF-8 编码，避免中文输出乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# WORKDIR 为当前工作目录，作为安全路径的根目录
WORKDIR = Path.cwd()

# 任务文件的存储目录（.tasks 文件夹）
TASKS_DIR = WORKDIR / ".tasks"

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

# 系统提示词，告诉模型它在哪个目录工作，以及使用任务工具来规划和追踪工作
SYSTEM = f"You are a coding agent at {WORKDIR}. Use task tools to plan and track work."


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


# -- TaskManager: 任务 CRUD 操作，带依赖图，持久化为 JSON 文件 --
class TaskManager:
    """
    任务管理器：负责创建、读取、更新、删除任务，
    并维护任务之间的依赖关系。
    任务以 JSON 文件形式存储在 .tasks/ 目录中，
    因此可以在上下文压缩中存活。
    """

    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir  # 任务文件存储目录
        self.dir.mkdir(exist_ok=True)  # 确保目录存在
        # 计算下一个可用的任务 ID（当前最大 ID + 1）
        self._next_id = self._max_id() + 1

    def _max_id(self) -> int:
        """
        获取当前最大的任务 ID。
        通过扫描 .tasks/ 目录下所有 task_*.json 文件来计算。
        例如 task_1.json 的 ID 就是 1。
        """
        # 遍历所有 task_*.json 文件，提取 ID
        # f.stem 是文件名不含扩展名（如 "task_1"），split("_")[1] 得到 "1"
        ids = [int(f.stem.split("_")[1]) for f in self.dir.glob("task_*.json")]
        return max(ids) if ids else 0  # 如果没有任务文件，返回 0

    def _load(self, task_id: int) -> dict:
        """
        从磁盘加载指定 ID 的任务。

        Args:
            task_id: 任务 ID

        Returns:
            任务字典
        """
        path = self.dir / f"task_{task_id}.json"  # 构造文件路径
        if not path.exists():
            raise ValueError(f"Task {task_id} not found")  # 文件不存在则报错
        return json.loads(path.read_text(encoding="utf-8"))  # 读取并解析 JSON

    def _save(self, task: dict) -> None:
        """
        将任务保存到磁盘（JSON 文件）。

        Args:
            task: 任务字典
        """
        path = self.dir / f"task_{task['id']}.json"  # 构造文件路径
        # 写入 JSON 文件，indent=2 让文件易读，ensure_ascii=False 保留中文
        path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")

    def create(self, subject: str, description: str = "") -> str:
        """
        创建新任务。

        Args:
            subject: 任务主题/标题（必填）
            description: 任务详细描述（可选）

        Returns:
            返回创建的任务的 JSON 字符串
        """
        # 构建任务字典
        task = {
            "id": self._next_id,          # 分配新 ID
            "subject": subject,            # 任务主题
            "description": description,    # 详细描述
            "status": "pending",           # 默认状态为 pending（待处理）
            "blockedBy": [],               # 默认没有被阻塞的任务（依赖列表）
            "owner": "",                   # 所有者（暂未使用）
        }
        self._save(task)                   # 保存到磁盘
        self._next_id += 1                 # 下一个任务的 ID +1
        return json.dumps(task, indent=2, ensure_ascii=False)

    def get(self, task_id: int) -> str:
        """
        获取指定任务 ID 的详细信息。

        Args:
            task_id: 任务 ID

        Returns:
            任务详情的 JSON 字符串
        """
        return json.dumps(self._load(task_id), indent=2, ensure_ascii=False)

    def update(
        self,
        task_id: int,
        status: str = None,
        addBlockedBy: list[int] = None,
        removeBlockedBy: list[int] = None,
    ) -> str:
        """
        更新任务的状态或依赖关系。

        Args:
            task_id: 任务 ID
            status: 新状态（pending/in_progress/completed）
            addBlockedBy: 添加阻塞此任务的任务 ID 列表
            removeBlockedBy: 从阻塞列表中移除的任务 ID 列表

        Returns:
            更新后任务的 JSON 字符串
        """
        task = self._load(task_id)  # 先加载现有任务

        # 更新状态
        if status:
            # 验证状态值是否合法
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Invalid status: {status}")
            task["status"] = status
            # 如果任务标记为完成，清除所有依赖（该任务从其他任务的 blockedBy 中移除）
            if status == "completed":
                self._clear_dependency(task_id)

        # 添加阻塞任务（当前任务被这些任务阻塞）
        if addBlockedBy:
            # set() 用于去重，再转回列表
            task["blockedBy"] = list(set(task["blockedBy"] + addBlockedBy))

        # 移除阻塞任务（解除对这些任务的依赖）
        if removeBlockedBy:
            # 从列表中过滤掉要移除的 ID
            task["blockedBy"] = [x for x in task["blockedBy"] if x not in removeBlockedBy]

        self._save(task)  # 保存更新后的任务
        return json.dumps(task, indent=2, ensure_ascii=False)

    def _clear_dependency(self, completed_id: int) -> None:
        """
        清除已完成任务的依赖关系。

        当一个任务完成时，需要从所有其他任务的 blockedBy 列表中移除它。
        这是一个"后向清理"操作，确保依赖该任务的其他任务可以继续执行。

        Args:
            completed_id: 已完成的任务 ID
        """
        # 遍历所有任务文件
        for task_file in self.dir.glob("task_*.json"):
            task = json.loads(task_file.read_text(encoding="utf-8"))
            # 如果该任务的 blockedBy 列表中包含 completed_id，则移除它
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
            key=lambda f: int(f.stem.split("_")[1]),
        )
        for task_file in files:
            tasks.append(json.loads(task_file.read_text(encoding="utf-8")))

        if not tasks:
            return "No tasks."

        lines = []
        for task in tasks:
            # 根据状态选择标记符号：[ ] 待处理, [>] 进行中, [x] 已完成
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }.get(task["status"], "[?]")  # 未知状态显示 [?]
            # 如果有阻塞任务，显示依赖信息
            blocked = f" (blocked by: {task['blockedBy']})" if task.get("blockedBy") else ""
            # 格式化输出：标记 ID 主题 依赖
            lines.append(f"{marker} #{task['id']}: {task['subject']}{blocked}")

        return "\n".join(lines)


# 全局任务管理器实例，供工具处理器使用
TASKS = TaskManager(TASKS_DIR)


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
    执行 Windows cmd 命令的封装。

    包含安全检查，阻止危险命令的执行。
    返回命令的标准输出和标准错误的合并内容。
    """
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]  # 危险命令黑名单
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 使用 shell=True 让 Windows 解析命令
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,  # 在指定目录执行命令
            capture_output=True,  # 捕获输出
            text=True,  # 返回文本而非字节
            timeout=120,  # 120 秒超时
        )
        # 合并 stdout 和 stderr
        output = (result.stdout + result.stderr).strip()
        # 截断过长的输出
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
        result = "\n".join(lines)
        return result[:50000]
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
            "description": "Run a Windows cmd-compatible shell command.",
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
            "name": "task_create",
            "description": "Create a new task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "description": {"type": "string"}},
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_update",
            "description": "Update a task's status or dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                    },
                    "addBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"}},
                    "removeBlockedBy": {
                        "type": "array",
                        "items": {"type": "integer"}},
                },
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_list",
            "description": "List all tasks with status summary.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_get",
            "description": "Get full details of a task by ID.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
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
    # 任务相关工具
    "task_create": lambda **kw: TASKS.create(kw["subject"], kw.get("description", "")),
    "task_update": lambda **kw: TASKS.update(
        kw["task_id"],
        kw.get("status"),
        kw.get("addBlockedBy"),
        kw.get("removeBlockedBy"),
    ),
    "task_list": lambda **kw: TASKS.list_all(),
    "task_get": lambda **kw: TASKS.get(kw["task_id"]),
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

    流程：
    1. 调用 LLM 生成回复
    2. 处理工具调用
    3. 收集工具结果并追加到消息历史
    4. 重复直到没有工具调用（LLM 直接回复文本）
    """
    while True:
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
            query = input("\033[36mrepro-s07 >> \033[0m")
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
