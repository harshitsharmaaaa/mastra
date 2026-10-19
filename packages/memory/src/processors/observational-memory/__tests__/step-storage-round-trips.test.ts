import { convertArrayToReadableStream, MockLanguageModelV2 } from '@internal/ai-sdk-v5/test';
import { Agent } from '@mastra/core/agent';
import type { AgentGenerateOptions } from '@mastra/core/agent';
import { InMemoryStore } from '@mastra/core/storage';
import { createTool } from '@mastra/core/tools';
import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

import { Memory } from '../../../index';

/**
 * Regression coverage for #27319 — with every observation threshold disabled
 * (high token limits, no buffered observation), each model call in a tool loop
 * was gated behind redundant serial storage work:
 *
 *  - MessageHistory.persistMessages re-read the thread it is about to write to
 *  - filterObservedMessages re-read the thread for the OM lastObservedMessageCursor
 *  - the OM pending-token write ran and was awaited on the model-call path
 *  - WorkingMemoryStateProcessor re-read resource working memory on every step
 *
 * This spec aims thresholds strictly above the run content so observer and
 * reflector never fire, drives a multi-step tool loop through the real
 * Memory + Agent stack, and counts memory-domain storage operations between
 * consecutive model calls.
 */

const FINAL_TEXT = 'All done. Here is the summary you asked for.';

type OpLog = { ops: string[]; modelCalls: number };

const MODEL_MARKER = '__modelCall__';

function createLoopingActorModel(log: OpLog, toolRounds: number) {
  let streamCallCount = 0;
  let generateCallCount = 0;

  const toolCallStream = (call: number) =>
    convertArrayToReadableStream([
      { type: 'stream-start' as const, warnings: [] },
      { type: 'response-metadata' as const, id: `r${call}`, modelId: 'mock', timestamp: new Date() },
      { type: 'tool-input-start' as const, id: `call-${call}`, toolName: 'test' },
      { type: 'tool-input-delta' as const, id: `call-${call}`, delta: JSON.stringify({ action: 'go' }) },
      { type: 'tool-input-end' as const, id: `call-${call}` },
      {
        type: 'finish' as const,
        finishReason: 'tool-calls' as const,
        usage: { inputTokens: 50, outputTokens: 20, totalTokens: 70 },
      },
    ]);

  const finalStream = () =>
    convertArrayToReadableStream([
      { type: 'stream-start' as const, warnings: [] },
      { type: 'response-metadata' as const, id: 'final', modelId: 'mock', timestamp: new Date() },
      { type: 'text-start' as const, id: 't1' },
      { type: 'text-delta' as const, id: 't1', delta: FINAL_TEXT },
      { type: 'text-end' as const, id: 't1' },
      {
        type: 'finish' as const,
        finishReason: 'stop' as const,
        usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      },
    ]);

  const toolCallGenerate = (call: number) => ({
    rawCall: { rawPrompt: null, rawSettings: {} },
    finishReason: 'tool-calls' as const,
    usage: { inputTokens: 50, outputTokens: 20, totalTokens: 70 },
    text: '',
    content: [
      {
        type: 'tool-call' as const,
        toolCallId: `call-${call}`,
        toolName: 'test',
        input: JSON.stringify({ action: 'go' }),
      },
    ],
    warnings: [],
  });

  const finalGenerate = () => ({
    rawCall: { rawPrompt: null, rawSettings: {} },
    finishReason: 'stop' as const,
    usage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
    text: FINAL_TEXT,
    content: [{ type: 'text' as const, text: FINAL_TEXT }],
    warnings: [],
  });

  return new MockLanguageModelV2({
    doStream: async () => {
      log.modelCalls++;
      log.ops.push(MODEL_MARKER);
      return { stream: streamCallCount < toolRounds ? toolCallStream(++streamCallCount) : finalStream() };
    },
    doGenerate: async () => {
      log.modelCalls++;
      log.ops.push(MODEL_MARKER);
      return generateCallCount < toolRounds ? toolCallGenerate(++generateCallCount) : finalGenerate();
    },
  });
}

/** Must never be invoked during these tests — surfaces any call loudly. */
function createNeverRunModel(log: OpLog, name: string) {
  return new MockLanguageModelV2({
    doStream: async () => {
      log.modelCalls++;
      log.ops.push(`${name}RAN`);
      throw new Error(`${name} ran although the observation thresholds were never reached`);
    },
  });
}

function createSimpleTool() {
  return createTool({
    id: 'test',
    description: 'Trigger tool for OM step-loop testing',
    inputSchema: z.object({ action: z.string().optional() }),
    execute: async () => ({ success: true }),
  });
}

type GatedWrite = { recordId: string; pendingMessageTokens: number };

function instrumentMemoryStore(store: InMemoryStore, gatedWrites: GatedWrite[]) {
  const resolvers: Array<() => void> = [];
  let gated = true;
  let released = false;
  const releaseAll = () => {
    released = true;
    gated = false;
    for (const resolve of resolvers.splice(0)) resolve();
  };

  const instrument = async (method: string) => {
    const memoryStore = await store.getStore('memory');
    const impl = (memoryStore as any)[method].bind(memoryStore);
    if (method === 'setPendingMessageTokens') {
      vi.spyOn(memoryStore as any, method).mockImplementation(async (...args: any[]) => {
        const value = args.at(-1) as number;
        if (gated) await new Promise<void>(resolve => resolvers.push(resolve));
        gatedWrites.push({ recordId: String(args[0] ?? args[1] ?? ''), pendingMessageTokens: value });
        return impl(...args);
      });
      return;
    }
    vi.spyOn(memoryStore as any, method).mockImplementation(async (...args: any[]) => {
      ops.push(method);
      return impl(...args);
    });
  };

  let ops: string[] = [];
  const instrumentAll = (...methods: string[]) => Promise.all(methods.map(method => instrument(method)));
  return { setOps: (log: OpLog) => (ops = log.ops), instrumentAll, releaseAll, isReleased: () => released };
}

function countsBetween(ops: string[], start: number, end: number, method: string): number {
  return ops.slice(start, end).filter(op => op === method).length;
}

function modelCallPositions(ops: string[]): number[] {
  return ops.map((op, index) => (op === MODEL_MARKER ? index : -1)).filter(index => index >= 0);
}

const messageText = (message: any): string => {
  if (typeof message?.content?.content === 'string') return message.content.content;
  return message?.content?.parts?.map((part: any) => part.text ?? '').join('') ?? '';
};

describe('OM step storage round trips below observation thresholds (#27319)', () => {
  it('does not gate each model call behind redundant thread/pending-token storage work', async () => {
    const store = new InMemoryStore();
    const opLog: OpLog = { ops: [], modelCalls: 0 };
    const gatedWrites: GatedWrite[] = [];
    const instrumenter = instrumentMemoryStore(store, gatedWrites);

    const observationCalls: OpLog = { ops: [], modelCalls: 0 };
    const reflectionCalls: OpLog = { ops: [], modelCalls: 0 };

    const toolRounds = 3;
    const memory = new Memory({
      storage: store,
      options: {
        workingMemory: { enabled: true, scope: 'resource' },
        observationalMemory: {
          enabled: true,
          scope: 'thread',
          observation: {
            model: createNeverRunModel(observationCalls, 'observer') as any,
            messageTokens: 100000,
            manageWorkingMemory: true,
          },
          reflection: {
            model: createNeverRunModel(reflectionCalls, 'reflector') as any,
            observationTokens: 100000,
          },
        },
      },
    });

    const agent = new Agent({
      id: 'om-step-round-trips',
      name: 'om-round-trips',
      instructions: 'Use the test tool a few times, then finish.',
      model: createLoopingActorModel(opLog, toolRounds) as any,
      tools: { test: createSimpleTool() },
      memory,
    });

    const thread = 'om-round-trips-1';
    const resource = 'om-round-trips-resource';
    await memory.createThread({ threadId: thread, resourceId: resource });

    instrumenter.setOps(opLog);
    await instrumenter.instrumentAll(
      'getThreadById',
      'saveThread',
      'saveMessages',
      'getResourceById',
      'getObservationalMemory',
      'listMessages',
      'listMessagesByResourceId',
      'setPendingMessageTokens',
    );

    // While the pending-token write is gated shut the whole conversation must
    // still finish. If a step blocks on the write, this watchdog releases the
    // gate so the test can fail fast with a pointed message instead of timing
    // out at the full test timeout.
    let watchdogFired = false;
    const hangGuard = setTimeout(() => {
      watchdogFired = true;
      instrumenter.releaseAll();
    }, 8000);

    try {
      const response = await agent.generate('Hello, run the test tool a few times.', {
        memory: { thread, resource },
        maxSteps: 10,
      } as AgentGenerateOptions<any>);
      expect(watchdogFired, 'a model call was blocked by the gated pending token write').toBe(false);
      expect(response.text).toContain('All done');
      expect(opLog.modelCalls).toBe(toolRounds + 1);
      expect(observationCalls.modelCalls).toBe(0);
      expect(reflectionCalls.modelCalls).toBe(0);

      const positions = modelCallPositions(opLog.ops);
      expect(positions).toHaveLength(toolRounds + 1);

      const gapStats: Array<{ gap: number; getThreadById: number; getResourceById: number }> = [];
      for (let gap = 1; gap < positions.length; gap++) {
        gapStats.push({
          gap,
          getThreadById: countsBetween(opLog.ops, positions[gap - 1], positions[gap], 'getThreadById'),
          getResourceById: countsBetween(opLog.ops, positions[gap - 1], positions[gap], 'getResourceById'),
        });
      }

      // Between upper-consecutive model calls only the runtime's own signal
      // bookkeeping may remain: no persistMessages thread check, no
      // cursor-fallback re-read, and no working memory reload.
      expect(gapStats).toHaveLength(toolRounds);
      for (const { gap, getThreadById, getResourceById } of gapStats) {
        expect(
          getThreadById,
          `gap ${gap} re-read the thread ${getThreadById} times between model calls`,
        ).toBeLessThanOrEqual(1);
        expect(getResourceById, `gap ${gap} re-read resource working memory`).toBe(0);
      }

      // Working memory loads once per run at most (plus any turn-start system
      // message build) instead of once per med step.
      expect(opLog.ops.filter(op => op === 'getResourceById').length).toBeLessThanOrEqual(2);

      // ── Background write lands with the final value after unlock ──
      instrumenter.releaseAll();
      await memory.settled();

      const memoryStore = await store.getStore('memory');
      expect(gatedWrites.length).toBeGreaterThan(0);
      const record = await memoryStore!.getObservationalMemory(thread, resource);
      const lastWrite = gatedWrites.at(-1)!;
      expect(record?.pendingMessageTokens).toBe(lastWrite.pendingMessageTokens);
      expect(lastWrite.recordId).toBe(record?.id);

      // ── Conversation transcript intact ──
      const stored = await memoryStore!.listMessages({
        threadId: thread,
        format: 'v2',
        includeTotal: false,
        perPage: false,
      } as any);
      const storedMessages: any[] = stored.messages;
      expect(messageText(storedMessages[0])).toContain('Hello, run the test tool');
      expect(messageText(storedMessages.at(-1))).toContain('All done');
    } finally {
      clearTimeout(hangGuard);
      instrumenter.releaseAll();
    }
  }, 30000);
});
