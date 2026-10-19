---
'@mastra/memory': patch
---

Fixed slow agent turns when observational memory is enabled. Below every observation threshold, each model call in a tool loop was gated behind several extra storage round trips: message persistence re-read the thread it was about to write to, the observed-message cursor lookup re-read the thread on every step, working memory was re-read on every step, and the pending-token write was awaited on the model-call path.

Those reads are now reused for the duration of the turn, and the pending-token write no longer blocks the conversation. Turns with observational memory enabled start responding noticeably sooner, especially against storage with high latency.
