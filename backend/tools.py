"""Tools the voice assistant's LLM can call.

Register a tool with @tool, giving it a JSON-schema for its parameters. Tools may be
sync or async and should return something JSON-serializable (or a string). A tool that
declares a `session` parameter receives the current VoiceSession (not part of the schema).

Website actions are not listed here one by one: the assistant delegates a whole task to
the browser agent (backend/browser_agent.py), which has its own tools and skills. Pictures
and videos are made by ComfyUI workflows on the RunPod pod (backend/media.py). Posts for
Instagram and X are drafts (backend/studio.py), made by following a content skill
(skills/content/*.md), and saved to FlowAI through its API (backend/flowai.py).
"""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from datetime import datetime

from .config import ROOT, settings
from .skills import load_skills
from .reel import MOODS, MOTIONS
from .studio import EDIT_MODES, POSITIONS, SIZES, STYLES

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
    fn, schema = _REGISTRY[name]
    params = inspect.signature(fn).parameters
    arguments = {k: v for k, v in arguments.items() if k not in context}
    missing = [k for k in schema["function"]["parameters"].get("required", []) if arguments.get(k) in (None, "")]
    if missing:
        return json.dumps({"error": f"{name} needs {' and '.join(missing)}; nothing was done"})
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
               "search_workflow_templates", "add_media_workflow", "edit_area"}


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
            "draft": {"type": "integer", "description": "Make it for this post draft: the result replaces the "
                                                        "slide's picture, in the draft's shape. An editing workflow "
                                                        "edits the slide's current picture"},
            "slide": {"type": "integer", "description": "Which slide of the draft (default 1)"},
        },
        "required": ["workflow", "prompt"],
    },
)
async def generate_media(workflow: str, prompt: str, session, image: str | None = None,
                         options: dict | None = None, draft: int | None = None, slide: int | None = None, **extra):
    # Models sometimes put inputs like width at the top level instead of in options
    if isinstance(options, dict) or options is None:
        options = {**extra, **(options or {})}
    if draft not in (None, ""):
        return await session.studio.generate(draft, slide, workflow, prompt, image, options)
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


# --- Post drafts for Instagram and X (backend/studio.py) -------------------------

_DRAFT = {"type": "integer", "description": "Draft number (default: the latest)"}
_LOOK = {
    "position": {"type": "string", "enum": POSITIONS, "description": "Where on the picture (default bottom)"},
    "size": {"type": "string", "description": f"{', '.join(SIZES)}, or a size in pixels (default large); "
                                              "long texts shrink to fit"},
    "color": {"type": "string", "description": "Text colour: a name or hex, e.g. white or #ffd400 (default white)"},
    "style": {"type": "string", "enum": STYLES, "description": "shadow (default), outline, box (a panel behind "
                                                              "the text, best on busy pictures) or plain"},
    "box_color": {"type": "string", "description": "Colour of the panel (style box) or outline (default black)"},
    "font": {"type": "string", "description": "A font's name from the fonts by style in your instructions, e.g. "
                                              "playfair or caveat (default bold)"},
}
_POST = {
    "placement": {"type": "string", "description": "Instagram: feed (default), story or reel. X: post"},
    "aspect": {"type": "string", "description": "Picture shape, e.g. 4:5, 1:1, 9:16, 16:9 (default: the "
                                                "placement's best shape)"},
    "title": {"type": "string", "description": "Short name for the draft, e.g. Summer sale launch"},
    "caption": {"type": "string", "description": "The post's text, written for the platform"},
    "media": {"type": "array", "items": {"type": "integer"},
              "description": "Picture or video numbers, one slide each, in order. On a change it is the whole new "
                             "list: slides keep their texts by position"},
    "background": {"type": "string", "description": "Colour of slides without a picture, e.g. #0e3b2e"},
}

STUDIO_TOOLS = {"create_draft", "update_draft", "add_text", "edit_text", "remove_text", "undo_draft",
                "show_draft", "load_skill", "make_video"}


@tool(
    "load_skill",
    "Load the step-by-step instructions of a content skill. Do this before making that kind of post.",
    {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Skill name from the list"}},
        "required": ["name"],
    },
)
def load_skill(name: str):
    skills = load_skills(ROOT / settings.content_skills_dir)
    skill = skills.get(str(name).strip())
    if not skill:
        return {"error": f"there is no skill {name!r}. Skills: {', '.join(skills) or 'none'}"}
    return f"Skill {skill.name}:\n{skill.instructions}"


@tool(
    "create_draft",
    "Make a post draft for Instagram or X, in one go: caption, the words on its slides and the pictures to make. "
    "It is drawn and shown on the user's screen after every change, with the platform's checks.",
    {
        "type": "object",
        "properties": {
            "platform": {"type": "string", "enum": ["instagram", "x"]},
            **_POST,
            "texts": {"type": "array", "description": "Words to draw on the slides, as add_text takes them. With "
                                                      "from_draft they replace the copied texts",
                      "items": {"type": "object", "required": ["text"], "properties": {
                          "text": {"type": "string", "description": "The words"},
                          "slide": {"type": "integer", "description": "Slide number (default 1)"}, **_LOOK}}},
            "pictures": {"type": "array", "items": {"type": "string"},
                         "description": "One picture prompt per slide, slide 1 first: each is made in the draft's "
                                        "shape and goes in by itself when ready"},
            "workflow": {"type": "string", "description": "Workflow for the pictures (default: text to image)"},
            "from_draft": {"type": "integer", "description": "Copy this draft's pictures, texts and caption, e.g. "
                                                             "to adapt an Instagram post for X"},
        },
        "required": ["platform"],
    },
)
async def create_draft(platform: str, session, placement: str | None = None, aspect: str | None = None,
                       title: str | None = None, caption: str | None = None, media: list | None = None,
                       background: str | None = None, from_draft: int | None = None, texts: list | None = None,
                       pictures: list | None = None, workflow: str | None = None):
    return await session.studio.create(platform, placement, aspect, title, caption, media, from_draft, background,
                                       texts, pictures, workflow)


@tool(
    "update_draft",
    "Change a draft's caption, title, placement, shape, background or pictures. Only what you give changes.",
    {"type": "object", "properties": {"draft": _DRAFT, **_POST}},
)
async def update_draft(session, draft: int | None = None, placement: str | None = None, aspect: str | None = None,
                       title: str | None = None, caption: str | None = None, media: list | None = None,
                       background: str | None = None):
    return await session.studio.update(draft, placement, aspect, title, caption, media, background)


@tool(
    "add_text",
    "Put text on a slide of a draft: a headline, an offer, a call to action. It is drawn crisply and wrapped "
    "to fit; never ask the image model for words.",
    {
        "type": "object",
        "properties": {
            "draft": _DRAFT,
            "text": {"type": "string", "description": "The words; a line break starts a new line"},
            "slide": {"type": "integer", "description": "Slide number (default 1)"},
            **_LOOK,
        },
        "required": ["text"],
    },
)
async def add_text(text: str, session, draft: int | None = None, slide: int | None = None, **look):
    return await session.studio.add_text(draft, text, slide, **{k: v for k, v in look.items() if k in _LOOK})


@tool(
    "edit_text",
    "Change a text on a draft: its words, look, or slide. Only what you give changes.",
    {
        "type": "object",
        "properties": {
            "draft": _DRAFT,
            "text_id": {"type": "integer", "description": "The text's number on the draft"},
            "text": {"type": "string", "description": "New words"},
            "slide": {"type": "integer", "description": "Move it to this slide"},
            **_LOOK,
        },
        "required": ["text_id"],
    },
)
async def edit_text(text_id: int, session, draft: int | None = None, text: str | None = None,
                    slide: int | None = None, **look):
    return await session.studio.edit_text(draft, text_id, text, slide,
                                          **{k: v for k, v in look.items() if k in _LOOK})


@tool(
    "remove_text",
    "Take a text off a draft.",
    {
        "type": "object",
        "properties": {"draft": _DRAFT, "text_id": {"type": "integer", "description": "The text's number"}},
        "required": ["text_id"],
    },
)
async def remove_text(text_id: int, session, draft: int | None = None):
    return await session.studio.remove_text(draft, text_id)


@tool(
    "undo_draft",
    "Put a draft back the way it was before its last change. Call again to go further back.",
    {"type": "object", "properties": {"draft": _DRAFT}},
)
async def undo_draft(session, draft: int | None = None):
    return await session.studio.undo(draft)


@tool(
    "edit_area",
    "Change only the area of a slide's picture that the user painted over on screen; the rest of the picture "
    "stays exactly as it is. Use it whenever an area is painted and the user says what to do with it.",
    {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": list(EDIT_MODES),
                     "description": "remove: take it away and show what's behind. replace: put something else there "
                                    "(say what in prompt). change: alter what is there, e.g. its colour or material. "
                                    "improve: sharper, better lit, more realistic"},
            "prompt": {"type": "string", "description": "In English. replace: what goes there, e.g. 'a small cactus "
                                                        "in a terracotta pot'. change: the instruction, e.g. 'Make the "
                                                        "dress dark blue'. improve: optional, what to improve"},
            "draft": {"type": "integer", "description": "Default: the draft with the painted area"},
            "slide": {"type": "integer", "description": "Default: the painted slide"},
        },
        "required": ["mode"],
    },
)
async def edit_area(mode: str, session, prompt: str = "", draft: int | None = None, slide: int | None = None):
    return await session.studio.edit_area(draft, slide, mode, prompt)


@tool(
    "make_video",
    "Make a draft's pictures (one or several) into a short video: your voice reads a line per picture in the "
    "user's voice while each picture slowly zooms or pans, with captions, music and an end card with a call to "
    "action. It goes into a draft of its own (an Instagram reel or an X post), made in the background; asked "
    "again, of either draft, that same video is made again with the changes.",
    {
        "type": "object",
        "properties": {
            "draft": {"type": "integer", "description": "The draft with the pictures, or the video's own draft to "
                                                        "make it again (default: the latest)"},
            "lines": {"type": "array", "items": {"type": "string"},
                      "description": "What the voice says over each picture: exactly one entry per picture, in "
                                     "order (a one-picture post has one entry). An entry can be one to three short "
                                     "spoken sentences; each sentence gets its own camera move. Hook first, no "
                                     "hashtags or emoji, and no call to action (that's cta_line). Leave out to keep "
                                     "the last video's lines"},
            "cta": {"type": "string", "description": "Words on the end card's button, 1 to 3 words, e.g. 'Shop now', "
                                                     "'Order yours', 'Book a table'; empty for no end card"},
            "cta_line": {"type": "string", "description": "The call to action the voice says on the end card, e.g. "
                                                          "'Order yours today, the link is in our bio.'"},
            "music": {"type": "string", "enum": [*MOODS, "none"],
                      "description": "Background music: " + ", ".join(f"{k} ({v})" for k, v in MOODS.items())},
            "captions": {"type": "boolean", "description": "Show the spoken words (default true)"},
            "motion": {"type": "string", "enum": ["auto", "pan", *MOTIONS],
                       "description": "How the pictures move (default auto: a different move for each; pan: "
                                      "side to side, one way then the other)"},
            "placement": {"type": "string", "description": "Instagram: reel (default), story or feed. X: post"},
            "color": {"type": "string", "description": "The end card button's colour (default: the post's own)"},
        },
    },
)
async def make_video(session, draft: int | None = None, lines: list | None = None, cta: str | None = None,
                     cta_line: str | None = None, music: str | None = None, captions: bool | None = None,
                     motion: str | None = None, placement: str | None = None, color: str | None = None):
    return await session.make_video(draft, lines=lines, cta=cta, cta_line=cta_line, music=music, captions=captions,
                                    motion=motion, placement=placement, color=color)


@tool(
    "show_draft",
    "Show a draft again on the user's screen and get all its details. Without a number, lists the drafts.",
    {"type": "object", "properties": {"draft": {"type": "integer", "description": "Draft number"}}},
)
async def show_draft(session, draft: int | None = None):
    return await session.studio.show(draft)


# --- FlowAI: saving, finding and scheduling posts (backend/flowai.py) --------------

FLOWAI_TOOLS = {"save_draft", "find_posts", "open_post", "find_assets", "use_assets", "schedule_post",
                "find_campaigns", "open_campaign"}


@tool(
    "save_draft",
    "Save a draft to FlowAI as a post with status draft, or save its latest changes to the same post. The team "
    "sees it in FlowAI's Composer and calendar. A draft opened from a campaign is saved into that campaign.",
    {
        "type": "object",
        "properties": {
            "draft": _DRAFT,
            "account": {"type": "integer", "description": "FlowAI account number (default: the only account on "
                                                          "the draft's platform)"},
        },
    },
)
async def save_draft(session, draft: int | None = None, account: int | None = None):
    return await session.flowai.save(draft, account)


@tool(
    "find_posts",
    "Find posts saved in FlowAI (drafts, scheduled and published), newest first.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words in the title or caption"},
            "status": {"type": "string", "enum": ["draft", "scheduled", "published", "failed"]},
            "platform": {"type": "string", "enum": ["instagram", "x"]},
        },
    },
)
async def find_posts(session, query: str | None = None, status: str | None = None, platform: str | None = None):
    return await session.flowai.find_posts(query, status, platform)


@tool(
    "open_post",
    "Open a FlowAI post as a draft here, to show or change it. Saving the draft updates the same post.",
    {
        "type": "object",
        "properties": {"post": {"type": "integer", "description": "The post's number in FlowAI"}},
        "required": ["post"],
    },
)
async def open_post(post: int, session):
    return await session.flowai.open_post(post)


@tool(
    "find_campaigns",
    "Find the user's FlowAI campaigns, newest first: name, stage and how many posts each has.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Words in the campaign's name"}},
    },
)
async def find_campaigns(session, query: str | None = None):
    return await session.flowai.find_campaigns(query)


@tool(
    "open_campaign",
    "Bring a FlowAI campaign's posts into this conversation as drafts, one per account version, to show or "
    "change them. Saving one changes that post in the campaign; a changed version waits for the user's approval.",
    {
        "type": "object",
        "properties": {"campaign": {"type": "integer", "description": "The campaign's number from find_campaigns"}},
        "required": ["campaign"],
    },
)
async def open_campaign(campaign: int, session):
    return await session.flowai.open_campaign(campaign)


@tool(
    "find_assets",
    "Find pictures and videos in the FlowAI gallery, newest first.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Words in the file name"},
            "kind": {"type": "string", "enum": ["image", "video"]},
        },
    },
)
async def find_assets(session, query: str | None = None, kind: str | None = None):
    return await session.flowai.find_assets(query, kind)


@tool(
    "use_assets",
    "Bring pictures or videos from the FlowAI gallery into this conversation; each gets a number to use in drafts.",
    {
        "type": "object",
        "properties": {"assets": {"type": "array", "items": {"type": "integer"},
                                  "description": "Asset numbers from find_assets"}},
        "required": ["assets"],
    },
)
async def use_assets(assets: list, session):
    return await session.flowai.use_assets(assets)


@tool(
    "schedule_post",
    "Ask the user to approve publishing a draft at a time. It saves the draft first if needed, then shows an "
    "Approve button on screen. Nothing is booked until the user presses it; you are told when they do.",
    {
        "type": "object",
        "properties": {
            "draft": _DRAFT,
            "when": {"type": "string", "description": "Date and time in the user's time zone, e.g. 2026-10-16T09:00, "
                                                      "or \"queue\" for the account's next free posting time"},
        },
        "required": ["when"],
    },
)
async def schedule_post(when: str, session, draft: int | None = None):
    return await session.flowai.schedule(draft, when)
