"""Tools the voice assistant's LLM can call.

Register a tool with @tool, giving it a JSON-schema for its parameters. Tools may be
sync or async and should return something JSON-serializable (or a string). A tool that
declares a `session` parameter receives the current VoiceSession (not part of the schema).

Website actions are not listed here one by one: the assistant delegates a whole task to
the browser agent (backend/browser_agent.py), which has its own tools and skills.
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


def schemas(exclude: set[str] = frozenset()) -> list[dict]:
    return [schema for name, (_, schema) in _REGISTRY.items() if name not in exclude]


async def run(name: str, arguments: dict, **context) -> str:
    """`context` values (e.g. session=...) are passed to tools that declare them."""
    if name not in _REGISTRY:
        return json.dumps({"error": f"unknown tool {name!r}"})
    fn, _ = _REGISTRY[name]
    params = inspect.signature(fn).parameters
    arguments = {k: v for k, v in arguments.items() if k not in context}
    try:
        result = fn(**arguments, **{k: v for k, v in context.items() if k in params})
        if inspect.isawaitable(result):
            result = await result
    except Exception as exc:
        log.exception("tool %s failed", name)
        return json.dumps({"error": str(exc)})
    return result if isinstance(result, str) else json.dumps(result, default=str)


# --- Built-in tools -------------------------------------------------------------

@tool("get_current_time", "Get the current local date and time.")
def get_current_time():
    return {"now": datetime.now().strftime("%A %d %B %Y, %I:%M %p")}


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


# --- Website (delegated to the browser agent) -------------------------------------

BROWSER_TOOLS = {"browser_task", "browser_task_status", "cancel_browser_task"}


@tool(
    "browser_task",
    "Start a task on the website in the web browser: navigating, clicking, filling forms, "
    "or reading information from pages. It runs in the background and you are told the result "
    "when it finishes. Give a complete, self-contained instruction including every detail the "
    "user gave.",
    {
        "type": "object",
        "properties": {"instruction": {"type": "string", "description": "What to do, in plain language"}},
        "required": ["instruction"],
    },
)
async def browser_task(instruction: str, session):
    return await session.start_browser_task(instruction)


@tool("browser_task_status", "Check progress of the running browser task.")
def browser_task_status(session):
    return session.browser_task_status()


@tool("cancel_browser_task", "Stop the running browser task.")
async def cancel_browser_task(session):
    return await session.cancel_browser_task()
