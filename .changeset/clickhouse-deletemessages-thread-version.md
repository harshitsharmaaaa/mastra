---
'@mastra/clickhouse': patch
---

Fixed ClickHouse `deleteMessages` bumping thread `updatedAt` with an `ALTER TABLE ... UPDATE` mutation. The store now inserts a new thread version with a strictly newer `updatedAt` instead, matching the existing `ReplacingMergeTree` write pattern used by other thread operations. This avoids ClickHouse mutation limits on managed deployments and keeps the latest thread version unambiguous for `ORDER BY updatedAt DESC LIMIT 1` reads.
