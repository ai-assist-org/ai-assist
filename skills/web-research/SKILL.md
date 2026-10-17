---
name: web-research
description: Conduct comprehensive web research on a question and produce a cited report. Use when the user asks to "research", "investigate", "do a deep dive", "compare X vs Y", "find out about", or wants a thorough, multi-source answer with citations rather than a single quick lookup. Runs the search/read/synthesize loop in a background task so fetched pages stay out of the main conversation; returns a saved report. Reuses the built-in web_search + web_fetch tools (keyless by default; better with provider keys).
argument-hint: <research question>
---

# Web Research

Produce a thorough, **cited** answer to a research question by searching the web,
reading the best sources, and synthesizing a report — with the heavy reading done
in a background task so the main conversation stays clean.

Question to research: **$ARGUMENTS**

## How to run it

1. If the question in `$ARGUMENTS` is empty or vague, ask the user for a clear
   research question (and scope/time-frame) before proceeding.

2. Pick a short kebab-case report name from the topic, e.g. `research-<topic>`.

3. **Delegate the work to a background task** with `internal__run_background` so the
   many fetched pages never enter this conversation's context. Pass:
   - `description`: `"Web research: <topic>"`
   - `save_to_report`: the report name from step 2
   - `max_time_seconds`: `600` (research is multi-step)
   - `prompt`: the research brief below, with `$ARGUMENTS` filled in.

4. Tell the user the research is running in the background, name the report it will
   write, and that they'll be notified when it completes. Do **not** start fetching
   pages yourself in this conversation.

5. When the task completes, read the report with `internal__read_report` and give the
   user a short summary plus the report name. If they want more, point them at the
   full report.

> If background tasks are unavailable (e.g. non-interactive mode), run the research
> brief below inline instead, then save the result with `internal__write_report`.

## Research brief (the background prompt)

> You are researching: **$ARGUMENTS**
>
> 1. Use `internal__web_search` to find sources. Start broad, then run 2–4 refined
>    queries covering sub-questions and alternative phrasings. Prefer primary and
>    authoritative sources.
> 2. Use `internal__web_fetch` to read the full content of the ~5–8 most relevant
>    results (not just snippets). For pages that fail to fetch, try another source.
> 3. Cross-check claims across at least two independent sources. Note where sources
>    disagree, and flag anything you could not verify.
> 4. Synthesize a structured markdown report:
>    - **Summary** — the direct answer in a few sentences.
>    - **Findings** — organized by sub-topic, with inline citations like `[1]`.
>    - **Caveats / open questions** — gaps, disagreements, recency limits.
>    - **Sources** — numbered list of `[n] Title — URL` for every citation used.
> 5. Ground every non-obvious claim in a citation. Do not invent URLs or facts; if
>    the web results don't answer part of the question, say so explicitly.
> 6. Save the finished report (this happens automatically via `save_to_report`).

## Notes

- Works with no API keys (DuckDuckGo search + built-in HTTP fetch). Relevance and
  anti-bot coverage improve automatically when Tavily / Brave / Firecrawl keys are
  set — this skill is provider-agnostic.
- Use `__save_to_file` on `internal__web_fetch` for very large pages so they don't
  bloat even the background task's context.
