"""Quickstart: run against an OpenAI-compatible server you control.

Example with a local Ollama server (nothing leaves your machine):
    ollama pull llama3.1
    ODIN_BASE_URL=http://localhost:11434/v1 python examples/quickstart.py
"""

import os

from odin_agent import Message, OdinAgent, OdinClient, Role

MODEL = os.getenv("ODIN_MODEL", "llama3.1")


def main() -> None:
    client = OdinClient()  # reads ODIN_BASE_URL (required) and ODIN_API_KEY (optional)

    print("--- 1. Single completion ---")
    response = client.chat_complete(
        messages=[Message(role=Role.USER, content="Bir cümleyle kendini tanıt.")],
        model=MODEL,
    )
    print(response.message.content)

    print("\n--- 2. Agent with a local tool ---")
    agent = OdinAgent(model=MODEL, client=client, system_prompt="Sen bir stok asistanısın.")

    def check_stock(product_code: str) -> dict:
        """Demo data; replace with a lookup in your own system."""
        return {"product_code": product_code, "in_stock": True, "quantity": 42}

    agent.register_tool(
        name="check_stock",
        description="Verilen ürün kodunun stok adedini döndürür.",
        parameters={
            "type": "object",
            "properties": {"product_code": {"type": "string", "description": "Ürün kodu"}},
            "required": ["product_code"],
        },
        handler=check_stock,
    )
    print(agent.run("PRD-102 kodlu ürünün stoğu ne durumda?"))


if __name__ == "__main__":
    main()
