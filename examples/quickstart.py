"""Quickstart example demonstrating Odin Agent SDK usage."""

from odin_agent import OdinAgent, OdinClient, Message, Role


def main() -> None:
    # 1. Direct completion using OdinClient
    client = OdinClient(api_key="your_odin_api_key_here")

    print("--- 1. Direct Chat Completion ---")
    response = client.chat_complete(
        messages=[
            Message(role=Role.USER, content="Odin mimarisi hakkında kısa bilgi ver.")
        ],
        model="odin-v2-chat",
    )
    print("Response:", response.message.content)

    # 2. Autonomous Agent with custom tool registration
    print("\n--- 2. Autonomous Agent with Tools ---")
    agent = OdinAgent(
        name="InventoryBot",
        system_prompt="Sen Odin POS ve ERP sistemlerine entegre stok asistanısın.",
        client=client,
    )

    # Define a tool handler
    def check_stock(product_code: str) -> dict:
        """Simulated inventory lookup"""
        return {"product_code": product_code, "in_stock": True, "quantity": 42}

    # Register the tool
    agent.register_tool(
        name="check_stock",
        description="Verilen ürün kodunun güncel stok adedini sorgular.",
        parameters={
            "type": "object",
            "properties": {
                "product_code": {
                    "type": "string",
                    "description": "Sorgulanacak ürün barkod veya kodu",
                }
            },
            "required": ["product_code"],
        },
        handler=check_stock,
    )

    answer = agent.run("PRD-102 kodlu ürünün stoğunu kontrol et ve bilgi ver.")
    print("Agent Result:", answer)


if __name__ == "__main__":
    main()
