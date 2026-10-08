"""Browser sub-agent: completes one task on the website, one action per LLM turn.

The voice assistant hands it a plain-language instruction. Each turn the agent sees
the current page (see Browser.state), calls one or more action tools, and gets the
new page back. It ends by calling finish with a short result the assistant can say.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from .browser import Browser
from .llm import LLM
from .skills import Skill, catalogue

log = logging.getLogger(__name__)

StepCallback = Callable[[int, str, dict, str], Awaitable[None]]

PAGE_MARK = "\n\n--- Current page ---\n"
MAX_NUDGES = 2

SYSTEM_PROMPT = """You are a browser agent. You complete a task on a website for a user by \
operating a real web browser, one step at a time.

Each turn you receive the current page: its URL, the visible interactive elements with a \
number in brackets, and the page text. Act by calling tools, referring to elements by their \
number. Numbers change whenever the page changes, so only use numbers from the latest page.

Rules:
- If a skill below matches the task, call load_skill first and follow its steps.
- Prefer one action per turn, then look at the new page before deciding the next one.
- If an action fails or the page is not what you expected, look again and try another way. \
Some controls only appear after hovering their row.
- Before an irreversible action that the task did not explicitly ask for (paying, deleting \
data, sending a message), stop and call finish asking the user to confirm.
- If you need information you do not have (a password, a choice between options), call finish \
and say exactly what you need.
- When the task is done, call finish with a one or two sentence result in plain spoken \
language, including any information the user asked for. Do not claim success you have not \
seen on the page.

Website: {website}

Skills:
{skills}
"""


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []},
    }}


_ID = {"element": {"type": "integer", "description": "Element number from the latest page"}}

TOOLS = [
    _fn("click", "Click an element.", _ID, ["element"]),
    _fn("type", "Replace the text in a text field.", {
        **_ID,
        "text": {"type": "string"},
        "submit": {"type": "boolean", "description": "Press Enter afterwards (default false)"},
    }, ["element", "text"]),
    _fn("select", "Choose an option in a dropdown.", {**_ID, "option": {"type": "string"}}, ["element", "option"]),
    _fn("hover", "Move the mouse over an element (reveals hover-only controls).", _ID, ["element"]),
    _fn("press", "Press a keyboard key, e.g. Enter, Escape, Tab, ArrowDown.",
        {"key": {"type": "string"}}, ["key"]),
    _fn("scroll", "Scroll the page.", {"direction": {"type": "string", "enum": ["up", "down"]}}, ["direction"]),
    _fn("goto", "Open a URL.", {"url": {"type": "string"}}, ["url"]),
    _fn("back", "Go back to the previous page."),
    _fn("read_page", "Get the full text of the page when the excerpt in the page state is cut off."),
    _fn("load_skill", "Load the step-by-step instructions of a skill.",
        {"name": {"type": "string"}}, ["name"]),
    _fn("finish", "End the task and report the outcome to the user.",
        {"result": {"type": "string", "description": "One or two spoken sentences"}}, ["result"]),
]


class BrowserAgent:
    def __init__(self, browser: Browser, llm: LLM, skills: dict[str, Skill], website: str,
                 max_steps: int, on_step: StepCallback | None = None):
        self.browser = browser
        self.llm = llm
        self.skills = skills
        self.website = website
        self.max_steps = max_steps
        self.on_step = on_step

    async def run(self, task: str) -> str:
        system = SYSTEM_PROMPT.format(website=self.website or "(any)", skills=catalogue(self.skills))
        messages: list = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Task: {task}{PAGE_MARK}{await self.browser.state()}"},
        ]
        nudges = 0
        for step in range(1, self.max_steps + 1):
            t0 = time.perf_counter()
            msg = await self.llm.chat(messages, TOOLS)
            log.info("browser agent step %d: LLM %.0f ms", step, (time.perf_counter() - t0) * 1000)
            calls = msg.tool_calls or []
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": calls or None})

            if not calls:
                # The model answered in prose instead of acting
                if nudges < MAX_NUDGES:
                    nudges += 1
                    messages.append({"role": "user", "content": "Use a tool. If the task is complete, call finish."})
                    continue
                return msg.content or "I stopped without a result."

            for call in calls:
                name, args = call.function.name, dict(call.function.arguments or {})
                if name == "finish":
                    result = str(args.get("result") or msg.content or "Done.")
                    await self._report(step, name, args, result)
                    return result
                result = await self._execute(name, args)
                log.info("browser agent step %d: %s(%s) -> %s", step, name, args, result[:120])
                await self._report(step, name, args, result)
                messages.append({"role": "tool", "tool_name": name, "content": result})

            # Only the newest page state stays in context; older ones are dropped
            for m in messages:
                if isinstance(m, dict) and PAGE_MARK in m.get("content", ""):
                    m["content"] = m["content"].split(PAGE_MARK)[0] + "\n(old page state removed)"
            messages[-1]["content"] += PAGE_MARK + await self.browser.state()

        return f"I stopped after {self.max_steps} steps without finishing. The browser is at {self.browser.page.url}."

    async def _execute(self, name: str, args: dict) -> str:
        b = self.browser
        try:
            match name:
                case "click":
                    return await b.click(args["element"])
                case "type":
                    return await b.type(args["element"], args["text"], bool(args.get("submit", False)))
                case "select":
                    return await b.select(args["element"], args["option"])
                case "hover":
                    return await b.hover(args["element"])
                case "press":
                    return await b.press(args["key"])
                case "scroll":
                    return await b.scroll(args.get("direction", "down"))
                case "goto":
                    return await b.goto(args["url"])
                case "back":
                    return await b.back()
                case "read_page":
                    return await b.read_text()
                case "load_skill":
                    skill = self.skills.get(args.get("name", ""))
                    if not skill:
                        return f"Error: no skill named {args.get('name')!r}. Available: {', '.join(self.skills) or 'none'}"
                    return f"Skill {skill.name}:\n{skill.instructions}"
                case _:
                    return f"Error: unknown tool {name!r}"
        except (KeyError, ValueError, TypeError) as exc:
            return f"Error: bad arguments for {name}: {exc}"

    async def _report(self, step: int, name: str, args: dict, result: str):
        if self.on_step:
            try:
                await self.on_step(step, name, args, result)
            except Exception:
                log.exception("on_step callback failed")
