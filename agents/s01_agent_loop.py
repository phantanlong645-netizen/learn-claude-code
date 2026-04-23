#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

try:
    import readline
    # macOS 某些终端会用 libedit 代替 GNU readline，
    # 这里显式打开 UTF-8 相关设置，避免中文输入/退格异常。
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 从 .env 加载配置；override=True 表示本地练习时允许 .env 覆盖已有环境变量。
load_dotenv(override=True)

if os.getenv("ANTHROPIC_BASE_URL"):
    # 当使用兼容 Anthropic API 的第三方提供商时，
    # 清掉某些环境里遗留的 ANTHROPIC_AUTH_TOKEN，避免认证来源冲突。
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# client 负责和模型服务通信；
# MODEL 决定本次运行实际使用哪个模型。
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.environ["MODEL_ID"]

# SYSTEM 是系统提示词：
# 它告诉模型自己的角色、工作目录和工作方式。
# 这里刻意保持很短，只给最必要的约束。
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# TOOLS 是“模型可用动作”的声明，不是真正的执行逻辑。
# 模型只能先提出“我要调用 bash(command=...)”，
# 然后由下面的 Python 代码真正执行命令。
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    # s01 故意只保留一个非常小的安全护栏：
    # 拦住几个明显危险的命令，避免演示时误伤本机环境。
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # shell=True 让模型输出的普通 shell 命令可以直接执行。
        # cwd=os.getcwd() 表示所有命令都在当前项目目录下运行。
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        # 把 stdout/stderr 合并，是为了把“世界的反馈”统一返回给模型。
        # 对 agent 来说，报错信息和正常输出一样，都是下一步推理的观察。
        out = (r.stdout + r.stderr).strip()
        # 限制输出长度，防止一次命令把上下文塞爆。
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    # messages 是完整对话历史。
    # 这个列表会在循环中不断增长：
    # 用户消息 -> 助手响应 -> 工具结果 -> 助手响应 -> ...
    while True:
        # 1. 把当前历史 + 工具定义交给模型。
        # 如果模型认为需要行动，它会在 response.content 里返回 tool_use。
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 2. 先把“助手这一轮完整输出”追加进历史。
        # 这一步非常关键：不是只记文本，而是整个 response.content。
        # 因为里面可能同时包含文本块和 tool_use 块，
        # 后续继续对话时，模型需要看到自己刚才到底说了什么、调用了什么。
        messages.append({"role": "assistant", "content": response.content})

        # 3. stop_reason != "tool_use" 说明模型这轮已经不想继续调用工具，
        # 通常意味着它已经拿到了足够信息，可以直接结束。
        if response.stop_reason != "tool_use":
            return

        # 4. 如果模型请求了工具，就逐个执行工具调用，
        # 再把执行结果组装成 tool_result 喂回模型。
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # 这里只支持一个 bash 工具，所以直接取 command。
                # 真实系统里通常会有 dispatch map，把 tool name 映射到不同 handler。
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200])
                # tool_result 必须通过 tool_use_id 对应回本轮具体哪一次工具调用。
                # 模型下轮看到这个结果后，才知道“刚才那个动作得到了什么反馈”。
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})

        # 5. 工具结果要作为一条新的 user 消息塞回 messages。
        # 这是 Anthropic 工具调用协议的一部分：
        # 模型提出动作，外部世界返回观察，然后模型基于观察继续推理。
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    # history 是整个 REPL 会话共享的历史；
    # 连续提问时，后一个问题会继承前面的上下文。
    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        # 输入空字符串、q、exit 都直接退出演示。
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 每次用户输入先入历史，再进入 agent loop。
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # agent_loop 返回时，history[-1] 往往是最后一条 assistant 消息。
        # 如果其中包含文本块，这里就把文本打印到终端。
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()
