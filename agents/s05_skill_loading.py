#!/usr/bin/env python3
# Harness: on-demand knowledge -- domain expertise, loaded when the model asks.
"""
s05_skill_loading.py - Skills

两层的技能注入机制，避免系统提示膨胀：

    Layer 1 (便宜): skill 名称在系统提示中 (~100 tokens/skill)
    Layer 2 (按需): 完整技能内容在 tool_result 中

    skills/
      pdf/
        SKILL.md          <-- frontmatter (name, description) + body
      code-review/
        SKILL.md

    System prompt:
    +--------------------------------------+
    | You are a coding agent.              |
    | Skills available:                    |
    |   - pdf: Process PDF files...        |  <-- Layer 1: 仅元数据
    |   - code-review: Review code...      |
    +--------------------------------------+

    当模型调用 load_skill("pdf"):
    +--------------------------------------+
    | tool_result:                         |
    | <skill>                              |
    |   Full PDF processing instructions   |  <-- Layer 2: 完整内容
    |   Step 1: ...                        |
    |   Step 2: ...                        |
    | </skill>                            |
    +--------------------------------------+

关键洞察: "不要把所有东西都塞进系统提示，按需加载。"
"""

import os
import re
import subprocess
import yaml
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(override=True)

# 如果设置了 ANTHROPIC_BASE_URL，则移除 ANTHROPIC_AUTH_TOKEN（避免冲突）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# WORKDIR = 当前工作目录
WORKDIR = Path.cwd()
# 创建 Anthropic API 客户端（支持自定义 base_url）
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 从环境变量获取模型 ID
MODEL = os.environ["MODEL_ID"]
# skills 目录路径 = ./skills
SKILLS_DIR = WORKDIR / "skills"


# -- SkillLoader: 扫描 skills/<name>/SKILL.md 并解析 YAML frontmatter --
class SkillLoader:
    """技能加载器类，负责扫描和解析 SKILL.md 文件"""
    
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir  # skills 目录路径
        self.skills = {}  # 存储所有技能 {name: {meta, body, path}}
        self._load_all()  # 启动时加载所有技能

    def _load_all(self):
        """递归扫描 SKILL.md 文件并解析"""
        # 如果 skills 目录不存在，直接返回
        if not self.skills_dir.exists():
            return
        # 递归查找所有 SKILL.md 文件（rglob = recursive glob）
        for f in sorted(self.skills_dir.rglob("SKILL.md")):
            text = f.read_text()  # 读取文件内容
            meta, body = self._parse_frontmatter(text)  # 解析 frontmatter
            # 优先使用 YAML 中的 name，否则使用目录名
            name = meta.get("name", f.parent.name)
            # 存入字典：{技能名: {元数据, 正文, 文件路径}}
            self.skills[name] = {"meta": meta, "body": body, "path": str(f)}

    def _parse_frontmatter(self, text: str) -> tuple:
        """解析 YAML frontmatter（--- 包裹的部分）"""
        # 正则匹配：--- 开头，--- 结尾，中间是 YAML
        match = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
        # 如果没有 frontmatter，返回空字典和原文
        if not match:
            return {}, text
        try:
            # 解析 YAML 为 Python 字典
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            meta = {}
        # 返回 (元数据字典, 正文内容)
        return meta, match.group(2).strip()

    def get_descriptions(self) -> str:
        """Layer 1: 生成系统提示中的简短描述"""
        if not self.skills:
            return "(no skills available)"
        lines = []
        for name, skill in self.skills.items():
            # 从 meta 中获取 description
            desc = skill["meta"].get("description", "No description")
            # 从 meta 中获取 tags（可选）
            tags = skill["meta"].get("tags", "")
            # 格式：- skill名: 描述 [tags]
            line = f"  - {name}: {desc}"
            if tags:
                line += f" [{tags}]"
            lines.append(line)
        # 拼接成多行字符串
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        """Layer 2: 按需返回完整技能内容（在 tool_result 中）"""
        skill = self.skills.get(name)  # 尝试获取技能
        if not skill:
            # 技能不存在，返回错误信息
            return f"Error: Unknown skill '{name}'. Available: {', '.join(self.skills.keys())}"
        # 用 <skill> 标签包裹完整内容返回
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"


# 创建全局 SkillLoader 实例
SKILL_LOADER = SkillLoader(SKILLS_DIR)

# Layer 1: 将技能元数据注入系统提示
SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use load_skill to access specialized knowledge before tackling unfamiliar topics.

Skills available:
{SKILL_LOADER.get_descriptions()}"""


# -- 工具实现函数 --
def safe_path(p: str) -> Path:
    """安全路径检查：防止路径穿越攻击"""
    path = (WORKDIR / p).resolve()  # 转换为绝对路径
    # 检查是否在 WORKDIR 内
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """执行 bash/shell 命令"""
    # 危险命令黑名单
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 执行命令，捕获输出
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 限制输出长度
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: int = None) -> str:
    """读取文件内容"""
    try:
        lines = safe_path(path).read_text().splitlines()
        # 如果限制了行数，截断并显示剩余行数
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """写入文件内容"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件：替换指定文本"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        # 只替换第一次出现
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# 工具处理函数映射表（工具名 -> 处理函数）
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    # 新增：load_skill 工具 -> 调用 SkillLoader 获取内容
    "load_skill": lambda **kw: SKILL_LOADER.get_content(kw["name"]),
}

# 工具定义列表（暴露给模型的工具 schema）
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    # 新增 load_skill 工具定义
    {"name": "load_skill", "description": "Load specialized knowledge by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Skill name to load"}}, "required": ["name"]}},
]


def agent_loop(messages: list):
    """Agent 主循环：接收消息 -> 调用模型 -> 处理工具 -> 返回结果"""
    while True:
        # 调用 Anthropic Messages API
        response = client.messages.create(
            model=MODEL,          # 模型 ID
            system=SYSTEM,        # 系统提示（包含 Layer 1）
            messages=messages,   # 消息历史
            tools=TOOLS,          # 可用工具列表
            max_tokens=8000,     # 最大输出 tokens
        )
        # 将模型响应添加到消息历史
        messages.append({"role": "assistant", "content": response.content})
        # 如果不是因为调用工具而停止，说明是最终回复
        if response.stop_reason != "tool_use":
            return
        # 处理工具调用
        results = []
        # 遍历模型返回的所有内容块
        for block in response.content:
            if block.type == "tool_use":
                # 从映射表中获取处理函数
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    # 调用处理函数，传入参数
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                # 打印工具调用结果（用于调试）
                print(f"> {block.name}:")
                print(str(output)[:200])
                # 收集 tool_result，返还给模型
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(output)
                })
        # 将工具结果作为 user 消息添加，触发下一轮
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    """主入口：交互式命令行界面"""
    history = []  # 消息历史
    while True:
        try:
            # 读取用户输入
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        # q/exit/空行 退出
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 将用户消息加入历史
        history.append({"role": "user", "content": query})
        # 调用 agent 循环处理
        agent_loop(history)
        # 获取最后一条回复
        response_content = history[-1]["content"]
        # 如果是内容块列表，提取文本打印
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()