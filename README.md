# 🤖 Odin Agent SDK (Python)

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg?style=flat-square)](https://pypi.org/)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-red.svg?style=flat-square)](LICENSE)
[![Organization](https://img.shields.io/badge/Odin-Informatics-0F172A?style=flat-square&logo=github)](https://github.com/Odin-Informatics)

**The Official Python SDK for Odin AI Models, Intelligent Tool-Using Agents, and Enterprise RAG Pipelines.**

Developed by **[Odin Informatics](https://www.odinbilisim.com)** to provide lightweight, type-safe, and asynchronous building blocks for integrating Odin LLMs into mission-critical production workflows.

---

## ✨ Features

- ⚡ **Lightweight & High Performance:** Minimal dependencies built on `httpx` and `pydantic`.
- 🧠 **Autonomous Agent Loop:** Built-in multi-turn reasoning with dynamic tool registration and execution.
- 🌊 **Real-Time Token Streaming:** Low-latency SSE (Server-Sent Events) streaming support.
- 🛡️ **Type-Safe Models:** Native Pydantic v2 schemas for all messages, tools, completions, and token usages.
- 🏢 **Enterprise Ready:** First-class support for Odin POS, ERP, and database tool handlers.

---

## 📦 Installation

```bash
pip install odin-agent-sdk
```

Or install with development dependencies:

```bash
pip install "odin-agent-sdk[dev]"
```

---

## 🚀 Quickstart

### 1. Basic Chat Completion

```python
from odin_agent import OdinClient, Message, Role

client = OdinClient(api_key="your_odin_api_key_here")

response = client.chat_complete(
    messages=[
        Message(role=Role.USER, content="Explain the benefits of microservices architecture.")
    ],
    model="odin-v2-chat",
    temperature=0.7,
)

print(response.message.content)
```

### 2. Streaming Responses

```python
for token in client.stream_chat(
    messages=[Message(role=Role.USER, content="Write a quick Python sort script.")],
    model="odin-v2-chat",
):
    print(token, end="", flush=True)
```

### 3. Autonomous Tool-Calling Agent

```python
from odin_agent import OdinAgent

agent = OdinAgent(
    name="EnterpriseOpsBot",
    system_prompt="You are an intelligent operations agent for Odin cloud platforms.",
)

# Define your business tool
def query_server_metrics(region: str) -> dict:
    return {"region": region, "cpu_usage": "24%", "status": "healthy"}

# Register the tool
agent.register_tool(
    name="query_server_metrics",
    description="Fetches real-time server cluster metrics for a specific cloud region.",
    parameters={
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "e.g., tr-central-1, eu-west-1"}
        },
        "required": ["region"],
    },
    handler=query_server_metrics,
)

# Run the agent
result = agent.run("Check the health status of the tr-central-1 server cluster.")
print(result)
```

---

## ⚙️ Configuration & Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ODIN_API_KEY` | Your Odin AI Platform API Key | `""` |
| `ODIN_BASE_URL` | Base API Endpoint URL | `https://api.odinbilisim.com/v1` |

---

## 🤝 Contributing

Contributions from the open-source community are welcome! Please follow these guidelines:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Run linting and type checks (`ruff check .` && `mypy .`).
4. Commit your changes (`git commit -m 'feat: add amazing feature'`).
5. Push to your branch and open a Pull Request.

---

## 📄 License

Distributed under the **Apache-2.0 License**. See [LICENSE](LICENSE) for more information.

---

<div align="center">
<sub>Engineered with precision by <strong>Odin Informatics</strong> • <a href="https://www.odinbilisim.com">odinbilisim.com</a></sub>
</div>
