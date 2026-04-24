#!/usr/bin/env python3
"""
Repro for s05: Skill Loading

This follows the original s05 lesson closely:
1. Scan skills/*/SKILL.md files
2. Put only skill summaries into the system prompt (Layer 1)
3. Let the model call load_skill(name) to fetch full content on demand (Layer 2)

The only provider-specific change is the OpenAI-compatible tool-calling format.
"""

from __future__ import annotations

# inspect: 用来获取当前打印语句来自哪一行代码。
import inspect
# json: 解析模型返回的 tool arguments 字符串。
import json
# os: 读取环境变量。
import os
# re: 用正则解析 SKILL.md 里的 YAML frontmatter。
import re
# subprocess: 执行 bash/shell 命令。
import subprocess
# sys: 调整终端输出编码，避免 Windows GBK 报错。
import sys
# Path: 更安全地处理文件路径。
from pathlib import Path

# yaml: 解析 frontmatter 里的 YAML 元数据。
import yaml
# load_dotenv: 从 .env 加载环境变量。
from dotenv import load_dotenv
# OpenAI: OpenAI-compatible SDK 客户端，这里接的是 DashScope/MiniMax。
from openai import OpenAI


# 加载项目根目录 .env 中的环境变量。
load_dotenv(override=True)

# 强制标准输出用 utf-8，减少 Windows 终端编码报错。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 当前工作目录，也就是 agent 默认操作的根目录。
WORKDIR = Path.cwd()
# skills 目录位置，原版约定就是 ./skills。
SKILLS_DIR = WORKDIR / "skills"

# 创建 OpenAI-compatible 客户端。
client = OpenAI(
    # API key 从环境变量里读。
    api_key=os.environ["DASHSCOPE_API_KEY"],
    # DashScope OpenAI-compatible endpoint。
    base_url=os.getenv(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
# 默认模型名。
MODEL = os.getenv("MODEL_ID", "MiniMax-M2.5")


def log(*parts) -> None:
    # currentframe() 拿到当前函数自己的 frame。
    frame = inspect.currentframe()
    # f_back 跳到调用 log() 的那一行。
    caller = frame.f_back if frame else None
    # 组合成 [文件名:行号] 前缀，方便你定位 print 来源。
    location = f"[{Path(__file__).name}:{caller.f_lineno}]" if caller else f"[{Path(__file__).name}]"
    # 把所有参数拼成一个字符串。
    text = " ".join(str(part) for part in parts)
    # 真正打印。
    print(f"{location} {text}")


class SkillLoader:
    def __init__(self, skills_dir: Path):
        # 保存 skills 根目录。
        self.skills_dir = skills_dir
        # self.skills 的结构大概是：
        # {
        #   "pdf": {"meta": {...}, "body": "...", "path": "..."},
        #   "code-review": {...}
        # }
        self.skills: dict[str, dict] = {}
        # 初始化时就递归加载所有技能。
        self._load_all()

    def _load_all(self) -> None:
        # 如果 skills 目录不存在，就直接结束。
        if not self.skills_dir.exists():
            return

        # 递归查找所有 SKILL.md。
        for skill_file in sorted(self.skills_dir.rglob("SKILL.md")):
            # 读取技能文件全文。
            text = skill_file.read_text(encoding="utf-8")
            # 解析 frontmatter 和正文。
            meta, body = self._parse_frontmatter(text)
            # 优先用 frontmatter 里的 name，否则用目录名。
            name = meta.get("name", skill_file.parent.name)
            # 保存 skill 的元信息、正文和路径。
            self.skills[name] = {
                "meta": meta,
                "body": body,
                "path": str(skill_file),
            }

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        # 匹配这种格式：
        # ---
        # name: pdf
        # description: ...
        # ---
        # 正文内容...
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        # 如果没有 frontmatter，就把整个文件当正文。
        if not match:
            return {}, text

        try:
            # 把 frontmatter 解析成 Python dict。
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            # YAML 写坏了就兜底成空字典。
            meta = {}

        # 返回二元组：(元数据, 正文)
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        # Layer 1：给 system prompt 用的简短 skill 摘要。
        if not self.skills:
            return "(no skills available)"

        lines = []
        for name, skill in self.skills.items():
            # description 是 frontmatter 里的短描述。
            desc = skill["meta"].get("description", "No description")
            # tags 可选，有些 skill 会带标签。
            tags = skill["meta"].get("tags", "")
            # 生成一行类似：
            #   - pdf: Process PDF files [pdf, parse]
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        # 把多行 skill 摘要拼成一个字符串。
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        # Layer 2：按需返回完整技能正文。
        skill = self.skills.get(name)
        if not skill:
            # 如果 skill 不存在，返回错误和可用列表。
            available = ", ".join(self.skills.keys())
            return f"Error: Unknown skill '{name}'. Available: {available}"
        # 用 <skill> 标签包起来，方便模型识别“这是技能内容”。
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# 创建全局 SkillLoader，后面 system prompt 和 load_skill 都会用它。
SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1：系统提示里只放 skill 摘要，不放完整内容。
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


def safe_path(p: str) -> Path:
    # 把相对路径解析成绝对路径。
    path = (WORKDIR / p).resolve()
    # 限制路径不能逃出工作目录。
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    # 最简单的危险命令黑名单。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 在当前工作目录里执行 shell 命令。
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        # 合并 stdout 和 stderr。
        output = (result.stdout + result.stderr).strip()
        # 截断太长的输出。
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        # 先做安全路径检查，再读文件内容。
        lines = safe_path(path).read_text(encoding="utf-8").splitlines()
        # limit 存在时只返回前几行。
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        # 自动创建父目录。
        fp.parent.mkdir(parents=True, exist_ok=True)
        # 写入文件内容。
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        # 只允许替换存在的旧文本。
        if old_text not in content:
            return f"Error: Text not found in {path}"
        # 只替换第一次出现的 old_text。
        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 暴露给模型看的工具 schema。
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
            "name": "load_skill",
            # 这个工具不是做动作，而是按需加载完整技能知识。
            "description": "Load specialized knowledge by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name to load",
                    }
                },
                "required": ["name"],
            },
        },
    },
]


# 本地工具分发表：工具名 -> Python 实现。
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # load_skill 直接从 SkillLoader 取回完整技能正文。
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}


def build_assistant_message(message) -> dict:
    # 把 SDK message 对象转成可继续塞回 messages 的普通 dict。
    assistant_message = {
        "role": "assistant",
        "content": message.content or "",
    }

    tool_calls = message.tool_calls or []
    if tool_calls:
        # 如果模型请求了工具，把 tool_calls 结构也保留下来。
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
    # 找到工具对应的本地处理函数。
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Unknown tool: {tool_name}"

    try:
        # 用 arguments 作为关键字参数执行工具。
        return str(handler(**arguments))
    except Exception as e:
        return f"Error: {e}"


def agent_loop(messages: list[dict]) -> None:
    while True:
        # OpenAI-compatible API 需要把 system prompt 放进 messages 顶部。
        request_messages = [{"role": "system", "content": SYSTEM}, *messages]
        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        # 取第一条候选回复。
        message = response.choices[0].message

        # 先把 assistant 这轮消息记进历史。
        messages.append(build_assistant_message(message))

        if message.content:
            log(message.content)

        # 没有工具调用就说明这一轮结束了。
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return

        results = []
        for tool_call in tool_calls:
            # 当前工具名，例如 bash / load_skill。
            tool_name = tool_call.function.name
            try:
                # 先把模型返回的 JSON 字符串解析成 dict。
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as e:
                output = f"Error: Invalid tool arguments: {e}"
            else:
                # 再把参数交给本地工具执行。
                output = dispatch_tool(tool_name, arguments)

            # 打印工具日志。
            log(f"> {tool_name}:")
            log(output[:200])

            # 生成一条 role="tool" 消息，回填给下一轮模型。
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": output,
                }
            )

        # 这一轮可能有多个 tool result，所以用 extend。
        messages.extend(results)


if __name__ == "__main__":
    # history 保存整轮对话历史。
    history: list[dict] = []

    while True:
        try:
            # 交互式读取用户输入。
            query = input("\033[36mrepro-s05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # q / exit / 空输入时退出。
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 先把用户消息加入历史。
        history.append({"role": "user", "content": query})
        # 再运行 agent loop。
        agent_loop(history)
        # 打一个空行，方便观察多轮输出。
        log("")
