"""Tools the LLM can call. Website actions (Playwright) will be registered here later.

Register a tool with @tool, giving it a JSON-schema for its parameters. Tools may be
sync or async and should return something JSON-serializable (or a string).
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from datetime import datetime

log = logging.getLogger(__name__)

_REGISTRY: dict[str, tuple[Callable, dict]] = {}


def tool(name: str, description: str, parameters: dict | None = None):
    def wrap(fn: Callable) -> Callable:
        _REGISTRY[name] = (fn, {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}},
            },
        })
        return fn
    return wrap


def schemas() -> list[dict]:
    return [schema for _, schema in _REGISTRY.values()]


async def run(name: str, arguments: dict) -> str:
    if name not in _REGISTRY:
        return json.dumps({"error": f"unknown tool {name!r}"})
    fn, _ = _REGISTRY[name]
    try:
        result = fn(**arguments)
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        log.exception("tool %s failed", name)
        return json.dumps({"error": str(exc)})
    return result if isinstance(result, str) else json.dumps(result, default=str)


# --- Built-in tools -------------------------------------------------------------

@tool("get_current_time", "Get the current local date and time.")
def get_current_time():
    return {"now": datetime.now().strftime("%A %d %B %Y, %H:%M")}


@tool(
    "wait",
    "Pause for a number of seconds, e.g. while a page loads.",
    {
        "type": "object",
        "properties": {"seconds": {"type": "number", "description": "Seconds to wait (max 10)"}},
        "required": ["seconds"],
    },
)
async def wait(seconds: float):
    await asyncio.sleep(min(float(seconds), 10))
    return {"waited": seconds}
