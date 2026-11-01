"""
tools/browser.py — browser_action: real browser automation via
Playwright (pre-installed in the target environment this was built
against), for pages web_fetch can't handle (JS-rendered content,
clicking through a flow).

Gated (Tier 2, heavy) — see tools/__init__.py's GATED_TOOLS.

Scope/performance note: launches and closes a fresh headless Chromium
instance per call rather than keeping one browser alive across calls.
Simpler and safer (no shared-state lifecycle to manage, no risk of a
leaked browser process surviving a crashed request), at the cost of
roughly 1-2s of launch overhead per call. Keeping a persistent browser
instance alive across the process's lifetime is a reasonable future
optimization, not done here since it adds real lifecycle complexity
(when to close it, how a crash recovers) for a cost this tool's
expected call frequency doesn't obviously justify.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


class BrowserActionArgs(BaseModel):
    action: Literal["navigate_and_read", "click_and_read"] = Field(
        ..., description="navigate_and_read: load url and return its visible text. click_and_read: load url, click selector, then return the resulting page's visible text."
    )
    url: str = Field(..., description="URL to load.")
    selector: str | None = Field(None, description="CSS selector to click. Required for click_and_read.")
    timeout_ms: int = Field(15000, description="Per-step timeout in milliseconds (default 15000).")


async def browser_action(
    action: str,
    url: str,
    selector: str | None = None,
    timeout_ms: int = 15000,
) -> dict:
    if not _PLAYWRIGHT_AVAILABLE:
        return {"ok": False, "error": "playwright is not installed in this environment."}
    if action == "click_and_read" and not selector:
        return {"ok": False, "error": "selector is required for click_and_read."}

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page()
                await page.goto(url, timeout=timeout_ms)
                if action == "click_and_read":
                    await page.click(selector, timeout=timeout_ms)
                    await page.wait_for_load_state("networkidle", timeout=timeout_ms)
                text = await page.inner_text("body")
                final_url = page.url
            finally:
                await browser.close()
        if len(text) > 20_000:
            text = text[:20_000] + "\n\n[... truncated at 20 000 chars ...]"
        return {"ok": True, "url": final_url, "content": text}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def build_browser_tool() -> StructuredTool:
    return StructuredTool.from_function(
        coroutine=browser_action,
        name="browser_action",
        description=(
            "Load a URL in a real headless browser and return its visible text — for "
            "JS-rendered pages web_fetch can't read, optionally clicking one element first."
        ),
        args_schema=BrowserActionArgs,
    )
