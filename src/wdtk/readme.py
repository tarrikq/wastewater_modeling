"""Split the repository README into blocks the Streamlit overview page can render.

Streamlit renders Markdown differently from GitHub: it cannot load images or follow links by
relative path, and it treats every ``$`` as a math delimiter, so ``$639 to $924`` would render as
math. :func:`blocks` separates out the figures, points relative links at GitHub, and escapes the
dollar signs that are not math.
"""

import re
from typing import NamedTuple

REPO_URL = "https://github.com/tarrikq/water_distribution_modelling"
BRANCH = "master"
# The README's banner linking to the hosted app; redundant inside the app itself.
APP_BANNER = "**[Open the live dashboard]"
# The README's setup instructions and everything after them are for people with the repository.
_STOP_HEADING = re.compile(r"^##\s.*How to run\s*$")

_IMAGE_LINE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)\)\s*$")
_LINK = re.compile(r"(?<!!)\[(?P<text>[^\]]*)\]\((?P<target>[^)\s]+)\)")
_ABSOLUTE = re.compile(r"[a-z][a-z0-9+.-]*:|#")
# $$display$$, or $inline$ by GitHub's rule: no space just inside either delimiter and no digit
# right after the closing one. Currency such as "$412 to $736" fails the rule, so it is not math.
_MATH = re.compile(r"(\$\$.+?\$\$|\$(?=\S)[^$\n]+?(?<=\S)\$(?!\d))", re.DOTALL)


class Block(NamedTuple):
    """One renderable piece of the README."""

    kind: str  # "markdown" or "image"
    body: str  # Markdown text, or the image path relative to the repository root
    caption: str = ""


def github_url(path: str) -> str:
    """Return the GitHub URL of a repository-relative file or directory (trailing ``/``)."""
    kind = "tree" if path.endswith("/") else "blob"
    return f"{REPO_URL}/{kind}/{BRANCH}/{path.removeprefix('./')}"


def absolute_links(text: str) -> str:
    """Point every relative Markdown link at the file on GitHub."""

    def repl(m: re.Match[str]) -> str:
        target = m["target"]
        if _ABSOLUTE.match(target):
            return m[0]
        return f"[{m['text']}]({github_url(target)})"

    return _LINK.sub(repl, text)


def escape_dollars(text: str) -> str:
    """Escape every ``$`` outside a math span so Streamlit shows it literally."""
    parts = _MATH.split(text)  # odd indices are the math spans
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<!\\)\$", r"\\$", parts[i])
    return "".join(parts)


def blocks(text: str) -> list[Block]:
    """Split README text at its image lines into Markdown and image blocks.

    Text from the "How to run" heading onward is dropped.
    """
    out: list[Block] = []
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            out.append(Block("markdown", escape_dollars(absolute_links(body))))
        lines.clear()

    for line in text.splitlines():
        if _STOP_HEADING.match(line):
            break
        if line.startswith(APP_BANNER):
            continue
        image = _IMAGE_LINE.match(line)
        if image:
            flush()
            out.append(Block("image", image["path"], image["alt"]))
        else:
            lines.append(line)
    flush()
    return out
