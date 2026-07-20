---
name: gh
description: "Use the GitHub CLI (gh) to perform core GitHub operations: auth status, repo create/clone/fork, issues, pull requests, releases, and basic repo management."
---

# GitHub CLI (gh)

## Overview
Use `gh` for authenticated GitHub operations from the terminal.

## Quick checks
- Auth status: `gh auth status`
- Version: `gh --version`

## Core workflows

### Issues
- List: `gh issue list --limit 20`
- View: `gh issue view 42`

### Pull Requests
- List: `gh pr list`
- View: `gh pr view 123`
- View JSON: `gh pr view 123 --json mergeable --jq .mergeable`

### API
- Direct API call: `gh api repos/OWNER/REPO/pulls/NUMBER/comments`
- Reviews: `gh api repos/OWNER/REPO/pulls/NUMBER/reviews`
