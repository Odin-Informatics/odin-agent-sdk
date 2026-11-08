# odin-agent-sdk

[![CI](https://github.com/Odin-Informatics/odin-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Odin-Informatics/odin-agent-sdk/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A small, **experimental** Python library for building tool-calling agents on top of any
**OpenAI-compatible** `/chat/completions` endpoint — for example a local [Ollama](https://ollama.com)
or [vLLM](https://github.com/vllm-project/vllm) server.

> **Status:** experimental (0.x). The API may change. Not published on PyPI.
> Odin Bilişim does **not** operate a hosted model or public API for this SDK — you always point it at a
> server you run or trust. There is no default endpoint and the library sends no telemetry.

## What it does

- Sync and async clients with retries and exponential backoff (`OdinClient`, `AsyncOdinClient`)
- Text streaming (Server-Sent Events)
- A minimal agent loop (`OdinAgent`) that runs your Python functions when the model asks for them
- Conversation memory with a sliding window that keeps the system prompt
- Pydantic v2 models for messages, tool calls and token usage

## Install (from source)

```bash
git clone https://github.com/Odin-Informatics/odin-agent-sdk.git
cd odin-agent-sdk
pip install -e .            # or: pip install -e ".[dev]" for tests and linters
```

Python 3.10+ is required.

## Quickstart

```python
from odin_agent import Message, OdinAgent, OdinClient, Role

# Local Ollama server: nothing leaves your machine
client = OdinClient(base_url="http://localhost:11434/v1")

reply = client.chat_complete(
    messages=[Message(role=Role.USER, content="Bir cümleyle kendini tanıt.")],
    model="llama3.1",
)
print(reply.message.content)

agent = OdinAgent(model="llama3.1", client=client, system_prompt="Sen bir stok asistanısın.")

def check_stock(product_code: str) -> dict:
    return {"product_code": product_code, "quantity": 42}  # replace with your own lookup

agent.register_tool(
    name="check_stock",
    description="Returns the stock level of a product code.",
    parameters={"type": "object", "properties": {"product_code": {"type": "string"}}, "required": ["product_code"]},
    handler=check_stock,
)
print(agent.run("PRD-102 kodlu ürünün stoğu ne durumda?"))
```

Streaming:

```python
for token in client.stream_chat([Message(role=Role.USER, content="Merhaba")], model="llama3.1"):
    print(token, end="", flush=True)
```

A runnable version is in [`examples/quickstart.py`](examples/quickstart.py).

## Configuration

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `ODIN_BASE_URL` | Base URL of an OpenAI-compatible API, e.g. `http://localhost:11434/v1` | none — **required** (or pass `base_url=`) |
| `ODIN_API_KEY` | Bearer token, if your server needs one | empty (no `Authorization` header is sent) |

## Security notes

- Tools run **your** Python functions with arguments chosen by the model. Validate inputs inside each
  handler and never register tools that run shell commands, SQL or file operations without checks.
- Keep API keys in environment variables, not in code.

## Development

```bash
pip install -e ".[dev]"
ruff check . && mypy src/odin_agent --ignore-missing-imports && pytest --cov=odin_agent
```

Tests use `httpx.MockTransport`; they make no network calls.

## License

[Apache-2.0](LICENSE) · Maintained by [Odin Bilişim](https://www.odinbilisim.com), Tekirdağ.
