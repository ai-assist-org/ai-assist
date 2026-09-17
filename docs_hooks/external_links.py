"""MkDocs hook: rewrite links to repo-root files that live outside the site.

The guide pages under ``docs/`` link to sibling files at the repository root
(``../README.md``, ``../SECURITY.md``, ``../CONTRIBUTING.md``,
``../examples/README.md``). Those work on GitHub but are not part of the built
site, so we rewrite them to their GitHub URLs instead of shipping dead links.
The source Markdown files are left untouched.
"""

import re

_REPO_BLOB = "https://github.com/ai-assist-org/ai-assist/blob/main"

# Maps the root-relative link target to its GitHub blob path.
_TARGETS = {
    "../README.md": "/README.md",
    "../SECURITY.md": "/SECURITY.md",
    "../CONTRIBUTING.md": "/CONTRIBUTING.md",
    "../examples/README.md": "/examples/README.md",
}

# Matches ](../README.md) and ](../README.md#anchor)
_PATTERN = re.compile(r"\]\((\.\./[\w./-]+\.md)(#[\w-]+)?\)")


def on_page_markdown(markdown, **kwargs):  # noqa: ARG001
    def _replace(match):
        target, anchor = match.group(1), match.group(2) or ""
        blob = _TARGETS.get(target)
        if blob is None:
            return match.group(0)
        return f"]({_REPO_BLOB}{blob}{anchor})"

    return _PATTERN.sub(_replace, markdown)
