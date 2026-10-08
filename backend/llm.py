"""Ollama cloud chat client (GLM 5.3 Flash by default), streaming with tool calls."""

from collections.abc import AsyncIterator

from ollama import AsyncClient, Message


class LLM:
    def __init__(self, host: str, api_key: str, model: str, think: bool | str):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.client = AsyncClient(host=host, headers=headers)
        self.model = model
        self.think = think

    async def stream(self, messages: list, tools: list[dict]) -> AsyncIterator[Message]:
        response = await self.client.chat(
            model=self.model, messages=messages, tools=tools or None, stream=True, think=self.think
        )
        async for chunk in response:
            yield chunk.message
