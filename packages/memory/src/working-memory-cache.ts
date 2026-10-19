import { getMemoryRunState } from '@mastra/core/memory';
import type { RequestContext } from '@mastra/core/request-context';

/**
 * Cache key for resource-scoped working memory inside a run's `MemoryRunState`.
 *
 * The state-signal processor reads resource working memory through the
 * run-scoped cache so a multi-step turn reads it once instead of once per
 * step. Writers call `refreshWorkingMemoryCache` after updating the row so the
 * next read sees the new value.
 */
export function workingMemoryResourceCacheKey(resourceId: string): string {
  return `working-memory:resource:${resourceId}`;
}

/**
 * Drop the run-scoped working memory cache for a resource after its working
 * memory was written, so the next read in the same run re-reads the row.
 */
export function refreshWorkingMemoryCache(args: {
  requestContext?: RequestContext;
  memory: object;
  threadId?: string;
  resourceId?: string;
}): void {
  const { requestContext, memory, threadId, resourceId } = args;
  if (!requestContext || !resourceId) return;
  const runState = getMemoryRunState(requestContext, memory, threadId, resourceId);
  runState?.refresh(workingMemoryResourceCacheKey(resourceId));
}
