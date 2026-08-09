---
"@moonshot-ai/kimi-code": minor
---

Report token usage and context-window state over ACP: `kimi acp` now streams `usage_update` notifications while a turn runs and returns the turn's token totals on the prompt response, so ACP clients can show a live context meter and per-turn spend.
