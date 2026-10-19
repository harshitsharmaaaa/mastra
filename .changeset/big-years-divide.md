---
'@mastra/core': patch
---

Added an optional thread vouch to message persistence. A caller that already holds the thread row — for example the thread captured when a run starts — can pass it along and skip the extra read that checks whether the thread exists. Also added a way to drop a cached working memory value after a writer updates it, so a run-scoped cache never serves stale working memory.
