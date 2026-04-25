#!/usr/bin/env python3
"""
Repro for s06: Context Compact

该脚本演示了三层上下文压缩机制，用于在长对话中管理 token 用量：
1. Layer 1: micro_compact - 每轮对话后自动清理旧的工具结果
2. Layer 2: auto_compact - 当 token 估计超过阈值时自动压缩
3. Layer 3: manual compact - 手动调用压缩工具显式压缩

唯一针对特定 provider 的改动是 OpenAI 兼容的工具调用格式。
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 文件中的环境变量，override=True 表示覆盖系统已有的变量
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

# 自动压缩的 token 阈值，当估计 token 数超过此值时触发 auto_compact
THRESHOLD = 50000

# 存储对话 transcript（压缩前的历史记录）的目录
TRANSCRIPT_DIR = WORKDIR / ".transcripts"

# micro_compact 中保留的最近工具结果数量
KEEP_RECENT = 3

# micro_compact 中需要保留完整结果的工具名称集合（这些工具的结果不会被压缩）
PRESERVE_RESULT_TOOLS = {"read_file"}


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


def estimate_tokens(messages: list[dict]) -> int:
    """
    粗略估算消息列表的 token 数量。
    由于精确计算 token 需要特定模型的 tokenizer，这里用字符串长度除以 4 作为近似估计。
    这是一个保守的估算方式（通常一个 token 约等于 4 个字符）。
    """
    return len(str(messages)) // 4


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


def collect_tool_name_map(messages: list[dict]) -> dict[str, str]:
    """
    从消息历史中收集工具调用的 ID 到名称的映射。
    用于在 micro_compact 时根据 tool_call_id 查找对应的工具名称。
    """
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            # 将工具调用 ID 映射到函数名称
            tool_name_map[tool_call["id"]] = tool_call["function"]["name"]
    return tool_name_map


def micro_compact(messages: list[dict]) -> list[dict]:
    """
    Layer 1 压缩：微观压缩

    在每轮对话后自动执行，清理旧的工具结果消息，只保留最近的几条。
    这样可以持续控制上下文增长，而不需要完全压缩对话。
    """
    tool_results = []
    # 收集所有角色为 tool 的消息
    for msg_index, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        tool_results.append((msg_index, msg))

    # 如果工具结果数量在保留范围内，不做任何处理
    if len(tool_results) <= KEEP_RECENT:
        return messages

    # 构建工具名称映射表
    tool_name_map = collect_tool_name_map(messages)
    # 要清理的旧结果（除了最近 KEEP_RECENT 条之外的所有旧结果）
    to_clear = tool_results[:-KEEP_RECENT]

    # 遍历并清理旧的工具结果
    for _, result in to_clear:
        content = result.get("content", "")
        # 如果内容较短（<=100字符），不需要处理
        if not isinstance(content, str) or len(content) <= 100:
            continue

        tool_id = result.get("tool_call_id", "")  # 工具调用 ID
        # 获取工具名称：优先使用 result 中的 name 字段，否则从映射表查找
        tool_name = result.get("name") or tool_name_map.get(tool_id, "unknown")
        # 如果该工具在保留列表中，则不压缩其结果
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        # 将详细内容替换为简短的摘要描述
        result["content"] = f"[Previous: used {tool_name}]"

    return messages


def summarize_messages(messages: list[dict]) -> str:
    """
    调用 LLM 生成对话摘要。

    将对话历史发送给模型，要求其总结：
    1. 已完成的工作
    2. 当前状态
    3. 做出的关键决策

    这样可以在大幅压缩上下文的同时保留关键信息。
    """
    # 将消息转换为 JSON 字符串，取最后 80000 字符避免过长
    conversation_text = json.dumps(messages, ensure_ascii=False, default=str)[-80000:]
    # 调用模型生成摘要
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Summarize this conversation for continuity. Include: "
                    "1) What was accomplished, 2) Current state, 3) Key decisions made. "
                    "Be concise but preserve critical details.\n\n"
                    + conversation_text
                ),
            }
        ],
    )
    return response.choices[0].message.content or "No summary generated."


def auto_compact(messages: list[dict]) -> list[dict]:
    """
    Layer 2 压缩：自动压缩

    当 token 估计数量超过阈值时触发。
    将当前对话保存到 transcript 文件，然后用摘要替换整个对话历史。
    这样可以释放大量 token 空间，同时通过文件保存完整历史供后续参考。
    """
    # 确保 transcript 目录存在
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    # 使用时间戳创建唯一的 transcript 文件名
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"

    # 将每条消息写入 JSONL 文件（每行一条 JSON 记录）
    with open(transcript_path, "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False, default=str) + "\n")

    log(f"[transcript saved: {transcript_path}]")

    # 生成对话摘要
    summary = summarize_messages(messages)
    # 返回压缩后的消息列表，只包含摘要信息
    return [
        {
            "role": "user",
            "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}",
        }
    ]


def safe_path(p: str) -> Path:
    """
    安全路径解析，确保访问被限制在 WORKDIR 以内。

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
    # 危险命令黑名单
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
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
            "name": "compact",
            "description": "Trigger manual conversation compression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "description": "What to preserve in the summary"
                    }
                },
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
    "compact": lambda **kw: "Manual compression requested.",
}


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
    1. 执行 micro_compact 清理旧工具结果
    2. 检查是否需要 auto_compact
    3. 调用 LLM 生成回复
    4. 处理工具调用
    5. 收集工具结果并追加到消息历史
    6. 检查是否手动触发了 compact
    """
    while True:
        # Step 1: 微观压缩，清理旧的工具结果
        micro_compact(messages)

        # Step 2: 检查是否触发自动压缩
        if estimate_tokens(messages) > THRESHOLD:
            log("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        # Step 3: 构建请求消息，包含系统提示和完整对话历史
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
        # 标记是否手动触发了 compact
        manual_compact = False

        # 遍历处理每个工具调用
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                # 解析工具参数（JSON 字符串）
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                # 如果是 compact 工具，设置标记
                if tool_name == "compact":
                    manual_compact = True
                    output = "Compressing..."
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

        # 如果手动触发了 compact，执行压缩并结束本轮
        if manual_compact:
            log("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    # 初始化空的消息历史列表
    history: list[dict] = []

    # 主循环：持续读取用户输入
    while True:
        try:
            # 打印提示符，等待用户输入（带颜色的提示符）
            query = input("\033[36mrepro-s06 >> \033[0m")
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
