"""Tools the voice assistant's LLM can call.

Register a tool with @tool, giving it a JSON-schema for its parameters. Tools may be
sync or async and should return something JSON-serializable (or a string). A tool that
declares a `session` parameter receives the current VoiceSession (not part of the schema).

Website actions are not listed here one by one: the assistant delegates a whole task to
the browser agent (backend/browser_agent.py), which has its own tools and skills. Pictures
and videos are made by ComfyUI workflows on the RunPod pod (backend/media.py).
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


# --- Images and video (ComfyUI on the RunPod pod) --------------------------------

MEDIA_TOOLS = {"generate_media", "media_status", "cancel_media", "list_media_workflows",
               "search_workflow_templates", "add_media_workflow"}


@tool(
    "generate_media",
    "Create a picture or a video with a ComfyUI workflow, or edit or animate an existing picture. "
    "It runs in the background; the result appears on the user's screen and you are told when it is ready.",
    {
        "type": "object",
        "properties": {
            "workflow": {"type": "string", "description": "Workflow name from the list"},
            "prompt": {"type": "string", "description": "Detailed visual description in English: subject, "
                                                        "setting, style, lighting, camera, motion for videos"},
            "image": {"type": "string", "description": "Only for workflows that need an image: the number of a "
                                                       "picture in this conversation, e.g. \"3\" (default: the "
                                                       "latest), or an https URL"},
            "options": {"type": "object", "description": "Other workflow inputs, e.g. {\"width\": 1216, "
                                                         "\"height\": 832} or {\"length\": 81}"},
        },
        "required": ["workflow", "prompt"],
    },
)
async def generate_media(workflow: str, prompt: str, session, image: str | None = None,
                         options: dict | None = None, **extra):
    # Models sometimes put inputs like width at the top level instead of in options
    if isinstance(options, dict) or options is None:
        options = {**extra, **(options or {})}
    return await session.media.generate(workflow, prompt, image, options)


@tool("media_status", "Check progress of picture/video generations and workflow installs.")
def media_status(session):
    return session.media.status()


@tool(
    "cancel_media",
    "Stop a picture or video generation.",
    {
        "type": "object",
        "properties": {"job": {"type": "integer", "description": "Job number (default: the latest running)"}},
    },
)
async def cancel_media(session, job: int | None = None):
    return await session.media.cancel(job)


@tool("list_media_workflows", "List the ComfyUI workflows on the pod: what each makes and its inputs.")
async def list_media_workflows(session):
    return await session.media.list_workflows()


@tool(
    "search_workflow_templates",
    "Search ComfyUI's built-in workflow templates, to find one to add.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Keywords, e.g. \"wan image to video\""}},
        "required": ["query"],
    },
)
async def search_workflow_templates(query: str, session):
    return await session.media.search_templates(query)


@tool(
    "add_media_workflow",
    "Add a new workflow to the pod from a built-in template, a workflow file the user attached, or a URL "
    "of a workflow JSON. Installing its nodes and models runs in the background and can take minutes.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "New workflow name in lower_snake_case, e.g. wan14_i2v"},
            "description": {"type": "string", "description": "One sentence: what it makes and from what"},
            "template": {"type": "string", "description": "Template name from search_workflow_templates"},
            "file": {"type": "integer", "description": "Number of a workflow file the user attached"},
            "url": {"type": "string", "description": "https URL of a workflow JSON file"},
            "replace": {"type": "boolean", "description": "Overwrite an existing workflow with this name "
                                                          "(only if the user asked)"},
        },
        "required": ["name", "description"],
    },
)
async def add_media_workflow(name: str, session, description: str = "", template: str | None = None,
                             file: int | None = None, url: str | None = None, replace: bool = False):
    return await session.media.add_workflow(name, description, template, file, url, bool(replace))
