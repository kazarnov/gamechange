"""Skills: markdown instructions the browser agent loads when a task matches.

A skill is skills/<name>.md (or skills/<name>/SKILL.md):

    ---
    name: add-todos
    description: Add one or more items to the todo list.
    ---
    1. Click the "What needs to be done?" box.
    2. ...

Only the name and description are always shown to the agent; it loads the full
steps with load_skill when it decides to use one. Files are re-read for every task,
so edits apply without restarting the server. Files starting with "_" are ignored.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    instructions: str


def _parse(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8")
    meta, body = {}, text
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        meta = yaml.safe_load(front) or {}
    name = meta.get("name") or (path.parent.name if path.name == "SKILL.md" else path.stem)
    if not meta.get("description"):
        log.warning("skill %s has no description; skipping", path)
        return None
    return Skill(str(name), " ".join(str(meta["description"]).split()), body.strip())


def load_skills(directory: Path) -> dict[str, Skill]:
    skills = {}
    if not directory.is_dir():
        return skills
    for path in sorted([*directory.glob("*.md"), *directory.glob("*/SKILL.md")]):
        if path.name.startswith("_") or path.parent.name.startswith("_") or path.name == "README.md":
            continue
        try:
            skill = _parse(path)
        except Exception as exc:
            log.warning("could not load skill %s: %s", path, exc)
            continue
        if skill:
            skills[skill.name] = skill
    return skills


def catalogue(skills: dict[str, Skill]) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in skills.values()) or "(no skills defined)"
