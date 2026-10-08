"""Shared Playwright browser that the browser agent drives.

The LLM never sees HTML. state() numbers every visible interactive element on the
page (stored in a data-agent-id attribute) and returns a compact text listing such as

    [3] button "Add to cart"
    [4] textbox "Email" value=""

Actions then refer to elements by those numbers.
"""

import asyncio
import base64
import logging
import os
from pathlib import Path

from playwright.async_api import BrowserContext, Dialog, Page, Playwright, async_playwright
from playwright.async_api import Error as PlaywrightError

log = logging.getLogger(__name__)

MAX_ELEMENTS = 150
MAX_TEXT = 2500

SNAPSHOT_JS = r"""
(maxText) => {
  const SEL = 'a[href], button, input:not([type=hidden]), select, textarea, summary, ' +
    '[role=button], [role=link], [role=checkbox], [role=radio], [role=switch], [role=tab], ' +
    '[role=menuitem], [role=option], [role=combobox], [role=textbox], [role=searchbox], ' +
    '[contenteditable=""], [contenteditable=true], [onclick]';
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria;
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const t = by.split(/\s+/).map((id) => document.getElementById(id)?.innerText || '').join(' ');
      if (clean(t)) return t;
    }
    if (el.labels && el.labels.length) return el.labels[0].innerText;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) {
      if (['submit', 'button'].includes(el.type)) return el.value;
      return el.placeholder || el.title || el.name || '';
    }
    return el.innerText || el.title || el.querySelector('img[alt]')?.alt || '';
  };

  document.querySelectorAll('[data-agent-id]').forEach((e) => e.removeAttribute('data-agent-id'));
  const items = [];
  let n = 0;
  for (const el of document.querySelectorAll(SEL)) {
    const r = el.getBoundingClientRect();
    const st = getComputedStyle(el);
    if (r.width < 1 || r.height < 1 || st.visibility === 'hidden' || st.display === 'none') continue;
    const tag = el.tagName.toLowerCase();
    let kind = el.getAttribute('role') ||
      { a: 'link', button: 'button', select: 'select', textarea: 'textbox', summary: 'button' }[tag] || tag;
    if (tag === 'input') {
      kind = ['checkbox', 'radio', 'submit', 'button', 'file', 'range', 'date', 'color'].includes(el.type)
        ? el.type : 'textbox';
    }
    if (el.isContentEditable && !el.getAttribute('role')) kind = 'textbox';
    let name = clean(nameOf(el));
    // Controls in list/table rows ("Delete", "Toggle") are ambiguous on their own:
    // describe them by their row too. Unlabeled controls fall back to their parent.
    const row = el.closest('li, tr, [role=row], [role=listitem]') || (name ? null : el.parentElement);
    let context = clean(row?.innerText).slice(0, 60);
    if (context === name) context = '';
    const item = { id: ++n, kind, name: name.slice(0, 80), context };
    if (kind === 'textbox' || kind === 'combobox' || kind === 'searchbox') {
      item.value = clean(el.value ?? el.innerText).slice(0, 80);
    }
    if (tag === 'select') {
      item.value = clean(el.selectedOptions[0]?.text);
      item.options = [...el.options].slice(0, 20).map((o) => clean(o.text));
    }
    if (el.type === 'checkbox' || el.type === 'radio') item.checked = el.checked;
    const ariaChecked = el.getAttribute('aria-checked') ?? el.getAttribute('aria-selected');
    if (ariaChecked !== null) item.checked = ariaChecked === 'true';
    if (el.disabled || el.getAttribute('aria-disabled') === 'true') item.disabled = true;
    item.inView = r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth;
    el.setAttribute('data-agent-id', String(n));
    items.push(item);
  }
  const doc = document.documentElement;
  return {
    url: location.href,
    title: document.title,
    items,
    text: (document.body?.innerText || '').replace(/[ \t]+/g, ' ').replace(/\n\s*\n+/g, '\n').trim().slice(0, maxText),
    scrollY: Math.round(scrollY),
    viewport: innerHeight,
    height: doc.scrollHeight,
  };
}
"""


def _first_line(exc: Exception) -> str:
    return str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__


def _describe(item: dict) -> str:
    s = f"[{item['id']}] {item['kind']}"
    if item["name"]:
        s += f' "{item["name"]}"'
    if item.get("context"):
        s += f' in row "{item["context"]}"'
    if "value" in item:
        s += f' value="{item["value"]}"'
    if "options" in item:
        s += " options=" + "|".join(item["options"])
    if "checked" in item:
        s += " checked" if item["checked"] else " unchecked"
    if item.get("disabled"):
        s += " disabled"
    if not item["inView"]:
        s += " (off-screen)"
    return s


class Browser:
    """One browser window shared by all sessions; tasks take `lock` to use it."""

    def __init__(self, start_url: str, headless: bool, profile_dir: Path):
        self.start_url = start_url
        # No display (plain Linux server / WSL without WSLg): fall back to headless
        self.headless = headless or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        self.profile_dir = profile_dir
        self.lock = asyncio.Lock()
        self.pw: Playwright | None = None
        self.context: BrowserContext | None = None
        self._page: Page | None = None
        self.dialogs: list[str] = []

    async def start(self):
        self.pw = await async_playwright().start()
        # A persistent profile keeps cookies, so a login done once in the window sticks
        self.context = await self.pw.chromium.launch_persistent_context(
            str(self.profile_dir), headless=self.headless, viewport={"width": 1280, "height": 800},
        )
        self.context.on("page", self._on_new_page)
        page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        self._on_new_page(page)
        if self.start_url:
            await self.goto(self.start_url)
        log.info("Browser ready (%s) at %s", "headless" if self.headless else "visible", self.start_url)

    async def close(self):
        if self.context:
            await self.context.close()
        if self.pw:
            await self.pw.stop()

    def _on_new_page(self, page: Page):
        # Follow new tabs (target=_blank links, popups)
        self._page = page
        page.on("dialog", lambda d: asyncio.ensure_future(self._on_dialog(d)))

    async def _on_dialog(self, dialog: Dialog):
        self.dialogs.append(f"{dialog.type} dialog said {dialog.message!r} and was accepted")
        await dialog.accept()

    @property
    def page(self) -> Page:
        if self._page is None or self._page.is_closed():
            open_pages = [p for p in self.context.pages if not p.is_closed()]
            if not open_pages:
                raise RuntimeError("the browser has no open tab")
            self._page = open_pages[-1]
        return self._page

    # --- observation --------------------------------------------------------------

    async def state(self) -> str:
        """Text description of the page for the LLM; renumbers elements."""
        try:
            s = await self.page.evaluate(SNAPSHOT_JS, MAX_TEXT)
        except PlaywrightError as exc:  # mid-navigation; retry once things settle
            log.debug("snapshot failed (%s), retrying", _first_line(exc))
            await self._settle()
            s = await self.page.evaluate(SNAPSHOT_JS, MAX_TEXT)

        items = s["items"]
        if len(items) > MAX_ELEMENTS:  # keep what's on screen first
            items = sorted(items, key=lambda i: not i["inView"])[:MAX_ELEMENTS]
            items.sort(key=lambda i: i["id"])
        screens = max(1, round(s["height"] / max(s["viewport"], 1)))
        lines = [
            f"URL: {s['url']}",
            f"Title: {s['title']}",
            f"Scroll: {s['scrollY']}px of {s['height']}px (about {screens} screen(s) tall)",
        ]
        if self.dialogs:
            lines.append("Dialogs: " + "; ".join(self.dialogs))
            self.dialogs.clear()
        lines.append("Interactive elements:")
        lines += [_describe(i) for i in items] or ["(none)"]
        if len(s["items"]) > len(items):
            lines.append(f"... {len(s['items']) - len(items)} more off-screen elements not shown")
        lines += ["Page text:", s["text"] or "(empty)"]
        return "\n".join(lines)

    async def screenshot_b64(self) -> str:
        data = await self.page.screenshot(type="jpeg", quality=55)
        return base64.b64encode(data).decode()

    async def read_text(self) -> str:
        return (await self.page.inner_text("body"))[:12000]

    # --- actions (each returns a short result for the LLM) --------------------------

    def _el(self, element_id: int):
        return self.page.locator(f'[data-agent-id="{int(element_id)}"]').first

    async def _settle(self):
        """Gives navigations and client-side re-renders a moment to finish."""
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            await self.page.wait_for_load_state("networkidle", timeout=1500)
        except PlaywrightError:
            pass
        await asyncio.sleep(0.3)

    async def _act(self, description: str, action) -> str:
        try:
            await action()
        except PlaywrightError as exc:
            return f"Error: {_first_line(exc)}"
        await self._settle()
        return description

    async def goto(self, url: str) -> str:
        if "://" not in url:
            url = "https://" + url
        return await self._act(f"Opened {url}", lambda: self.page.goto(url, wait_until="domcontentloaded"))

    async def click(self, element_id: int) -> str:
        return await self._act(f"Clicked [{element_id}]", lambda: self._el(element_id).click(timeout=5000))

    async def hover(self, element_id: int) -> str:
        return await self._act(f"Hovering [{element_id}]", lambda: self._el(element_id).hover(timeout=5000))

    async def type(self, element_id: int, text: str, submit: bool = False) -> str:
        async def action():
            el = self._el(element_id)
            await el.fill(str(text), timeout=5000)
            if submit:
                await el.press("Enter")
        return await self._act(f"Typed into [{element_id}]" + (" and pressed Enter" if submit else ""), action)

    async def select(self, element_id: int, option: str) -> str:
        async def action():
            el = self._el(element_id)
            try:
                await el.select_option(label=option, timeout=3000)
            except PlaywrightError:
                await el.select_option(value=option, timeout=3000)
        return await self._act(f"Selected {option!r} in [{element_id}]", action)

    async def press(self, key: str) -> str:
        return await self._act(f"Pressed {key}", lambda: self.page.keyboard.press(key))

    async def scroll(self, direction: str) -> str:
        dy = {"up": -0.8, "down": 0.8}.get(direction, 0.8)
        return await self._act(
            f"Scrolled {direction}",
            lambda: self.page.evaluate("(f) => window.scrollBy(0, innerHeight * f)", dy),
        )

    async def back(self) -> str:
        return await self._act("Went back", lambda: self.page.go_back(wait_until="domcontentloaded"))
