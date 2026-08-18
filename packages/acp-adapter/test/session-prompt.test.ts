import { describe, expect, it } from 'vitest';

import {
  AgentSideConnection,
  ClientSideConnection,
  ndJsonStream,
  type Client,
  type ContentBlock,
  type ReadTextFileRequest,
  type ReadTextFileResponse,
  type RequestPermissionRequest,
  type RequestPermissionResponse,
  type SessionNotification,
  type WriteTextFileRequest,
  type WriteTextFileResponse,
} from '@agentclientprotocol/sdk';
import type { Event, KimiHarness, Session } from '@moonshot-ai/kimi-code-sdk';

import { AcpServer } from '../src/server';
import { AUTHED_STATUS } from './_helpers/harness-stubs';

class CollectingClient implements Client {
  readonly updates: SessionNotification[] = [];

  /**
   * Updates produced AFTER `session/new` returns. Phase 9.3 makes
   * `newSession` emit exactly one `available_commands_update` on
   * creation; tests in this file pre-date that emission and assert
   * only on prompt-driven updates, so we filter that variant out.
   */
  get promptUpdates(): readonly SessionNotification[] {
    return this.updates.filter(
      (n) =>
        (n.update as { sessionUpdate?: string }).sessionUpdate !==
        'available_commands_update',
    );
  }

  async requestPermission(_p: RequestPermissionRequest): Promise<RequestPermissionResponse> {
    throw new Error('CollectingClient.requestPermission should not be called in prompt test');
  }
  async sessionUpdate(n: SessionNotification): Promise<void> {
    this.updates.push(n);
  }
  async writeTextFile(_p: WriteTextFileRequest): Promise<WriteTextFileResponse> {
    throw new Error('CollectingClient.writeTextFile should not be called in prompt test');
  }
  async readTextFile(_p: ReadTextFileRequest): Promise<ReadTextFileResponse> {
    throw new Error('CollectingClient.readTextFile should not be called in prompt test');
  }
}

function makeInMemoryStreamPair(): {
  agentStream: ReturnType<typeof ndJsonStream>;
  clientStream: ReturnType<typeof ndJsonStream>;
} {
  const clientToAgent = new TransformStream<Uint8Array, Uint8Array>();
  const agentToClient = new TransformStream<Uint8Array, Uint8Array>();
  const agentStream = ndJsonStream(agentToClient.writable, clientToAgent.readable);
  const clientStream = ndJsonStream(clientToAgent.writable, agentToClient.readable);
  return { agentStream, clientStream };
}

/**
 * Construct a fake Session whose `prompt()` synchronously emits a
 * pre-recorded sequence of `Event`s through any subscribed listener.
 */
function makeScriptedSession(
  sessionId: string,
  script: readonly Event[],
): {
  session: Session;
  unsubscribeCount: () => number;
} {
  const listeners = new Set<(event: Event) => void>();
  let unsubCount = 0;
  const session = {
    id: sessionId,
    prompt: async (_input: unknown) => {
      // Emit asynchronously so the caller has time to set `settled`
      // before the first event lands (matches real RPC ordering).
      for (const ev of script) {
        for (const fn of listeners) fn(ev);
      }
    },
    cancel: async () => undefined,
    onEvent: (fn: (event: Event) => void) => {
      listeners.add(fn);
      return () => {
        unsubCount += 1;
        listeners.delete(fn);
      };
    },
  } as unknown as Session;
  return { session, unsubscribeCount: () => unsubCount };
}

const textBlock = (text: string): ContentBlock => ({ type: 'text', text });

describe('AcpServer session/prompt', () => {
  it('streams two AssistantDelta events as agent_message_chunk updates and resolves with end_turn', async () => {
    const sessionId = 'sess-A';
    const { session, unsubscribeCount } = makeScriptedSession(sessionId, [
      { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'hel' } as Event,
      { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'lo' } as Event,
      { type: 'turn.ended', sessionId, agentId: 'main', turnId: 1, reason: 'completed' } as Event,
    ]);
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const collecting = new CollectingClient();
    const client = new ClientSideConnection(() => collecting, clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });

    const response = await client.prompt({
      sessionId,
      prompt: [textBlock('hi')],
    });

    expect(response.stopReason).toBe('end_turn');

    // Give the agent side a tick to flush queued sessionUpdate writes
    // through the ndjson stream.
    await new Promise((resolve) => setTimeout(resolve, 20));

    expect(collecting.promptUpdates).toHaveLength(2);
    for (const note of collecting.promptUpdates) {
      expect(note.sessionId).toBe(sessionId);
    }
    const first = collecting.promptUpdates[0]?.update;
    const second = collecting.promptUpdates[1]?.update;
    expect(first).toMatchObject({
      sessionUpdate: 'agent_message_chunk',
      content: { type: 'text', text: 'hel' },
    });
    expect(second).toMatchObject({
      sessionUpdate: 'agent_message_chunk',
      content: { type: 'text', text: 'lo' },
    });

    // Listener must be unsubscribed exactly once after turn.ended fires.
    expect(unsubscribeCount()).toBe(1);
  });

  it('resolves with cancelled stopReason when turn.ended reason is cancelled', async () => {
    const sessionId = 'sess-B';
    const { session, unsubscribeCount } = makeScriptedSession(sessionId, [
      { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'partial' } as Event,
      { type: 'turn.ended', sessionId, agentId: 'main', turnId: 1, reason: 'cancelled' } as Event,
    ]);
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const collecting = new CollectingClient();
    const client = new ClientSideConnection(() => collecting, clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });

    const response = await client.prompt({
      sessionId,
      prompt: [textBlock('do something long')],
    });

    expect(response.stopReason).toBe('cancelled');
    expect(unsubscribeCount()).toBe(1);
  });

  it('rejects prompt with invalid_params when sessionId is unknown', async () => {
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => {
        throw new Error('createSession should not be called for unknown-id test');
      },
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const client = new ClientSideConnection(() => new CollectingClient(), clientStream);

    await expect(
      client.prompt({ sessionId: 'sess-does-not-exist', prompt: [textBlock('hi')] }),
    ).rejects.toMatchObject({ code: -32602 });
  });

  it('rejects prompt (and unsubscribes) when underlying session.prompt rejects', async () => {
    const sessionId = 'sess-C';
    const listeners = new Set<(event: Event) => void>();
    let unsubCount = 0;
    const session = {
      id: sessionId,
      prompt: async (_input: unknown) => {
        throw new Error('boom from session.prompt');
      },
      cancel: async () => undefined,
      onEvent: (fn: (event: Event) => void) => {
        listeners.add(fn);
        return () => {
          unsubCount += 1;
          listeners.delete(fn);
        };
      },
    } as unknown as Session;

    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const client = new ClientSideConnection(() => new CollectingClient(), clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });

    await expect(
      client.prompt({ sessionId, prompt: [textBlock('hi')] }),
    ).rejects.toBeDefined();
    expect(unsubCount).toBe(1);
  });

  it('rejects prompt when the SDK emits a turn.agent_busy error event', async () => {
    const sessionId = 'sess-busy';
    const { session, unsubscribeCount } = makeScriptedSession(sessionId, [
      {
        type: 'error',
        sessionId,
        agentId: 'main',
        code: 'turn.agent_busy',
        message: 'Cannot launch a new turn while another turn (ID 0) is active',
        details: { turnId: 0 },
        retryable: true,
      } as unknown as Event,
    ]);
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const client = new ClientSideConnection(() => new CollectingClient(), clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });

    await expect(
      client.prompt({ sessionId, prompt: [textBlock('hi')] }),
    ).rejects.toMatchObject({ code: -32600 });
    expect(unsubscribeCount()).toBe(1);
  });

  it('does not reject an already-started prompt when a later prompt gets busy', async () => {
    const sessionId = 'sess-busy-active';
    const listeners = new Set<(event: Event) => void>();
    let unsubCount = 0;
    let promptCall = 0;
    let firstError: unknown;
    let resolveFirstTurn: (() => void) | undefined;
    const firstTurn = new Promise<void>((resolve) => {
      resolveFirstTurn = () => {
        resolve();
      };
    });
    void firstTurn.then(() => {
      for (const fn of listeners) {
        fn({ type: 'turn.ended', sessionId, agentId: 'main', turnId: 1, reason: 'completed' } as Event);
      }
    });
    const session = {
      id: sessionId,
      prompt: async (_input: unknown) => {
        promptCall += 1;
        await Promise.resolve();
        if (promptCall === 1) {
          for (const fn of listeners) {
            fn({
              type: 'turn.started',
              sessionId,
              agentId: 'main',
              turnId: 1,
              origin: { kind: 'user' },
            } as unknown as Event);
          }
          await firstTurn;
          return;
        }
        for (const fn of listeners) {
          fn({
            type: 'error',
            sessionId,
            agentId: 'main',
            code: 'turn.agent_busy',
            message: 'Cannot launch a new turn while another turn (ID 1) is active',
            details: { turnId: 1 },
            retryable: true,
          } as unknown as Event);
        }
      },
      cancel: async () => undefined,
      onEvent: (fn: (event: Event) => void) => {
        listeners.add(fn);
        return () => {
          unsubCount += 1;
          listeners.delete(fn);
        };
      },
    } as unknown as Session;
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const client = new ClientSideConnection(() => new CollectingClient(), clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });

    const firstPrompt = client
      .prompt({ sessionId, prompt: [textBlock('active')] })
      .then(
        (response) => response,
        (error) => {
          firstError = error;
          throw error;
        },
      );
    await Promise.resolve();

    await expect(
      client.prompt({ sessionId, prompt: [textBlock('busy')] }),
    ).rejects.toMatchObject({ code: -32600 });
    expect(firstError).toBeUndefined();

    resolveFirstTurn?.();
    await expect(firstPrompt).resolves.toMatchObject({ stopReason: 'end_turn' });
    expect(unsubCount).toBe(2);
  });

  it('ignores a subagent turn.ended and resolves on the main agent turn.ended', async () => {
    const sessionId = 'sess-subagent';
    const { session, unsubscribeCount } = makeScriptedSession(sessionId, [
      { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'a' } as Event,
      { type: 'assistant.delta', sessionId, agentId: 'sub-1', turnId: 99, delta: 'leak' } as Event,
      { type: 'thinking.delta', sessionId, agentId: 'sub-1', turnId: 99, delta: 'leak' } as Event,
      {
        type: 'tool.call.started',
        sessionId,
        agentId: 'sub-1',
        turnId: 99,
        toolCallId: 'sub-tool',
        name: 'Shell',
        args: { command: 'echo leak' },
      } as Event,
      {
        type: 'tool.result',
        sessionId,
        agentId: 'sub-1',
        turnId: 99,
        toolCallId: 'sub-tool',
        output: 'leak',
      } as Event,
      // A subagent finishes its own turn while the main turn is still
      // running. Pre-fix this would resolve the parent prompt with
      // `end_turn` and leak the listener; post-fix it must be ignored.
      {
        type: 'turn.ended',
        sessionId,
        agentId: 'sub-1',
        turnId: 99,
        reason: 'completed',
      } as Event,
      { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'b' } as Event,
      { type: 'turn.ended', sessionId, agentId: 'main', turnId: 1, reason: 'completed' } as Event,
    ]);
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const collecting = new CollectingClient();
    const client = new ClientSideConnection(() => collecting, clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });

    const response = await client.prompt({
      sessionId,
      prompt: [textBlock('hi')],
    });

    expect(response.stopReason).toBe('end_turn');
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(collecting.promptUpdates).toHaveLength(2);
    expect(unsubscribeCount()).toBe(1);
  });
});

/**
 * `agent.status.updated` carries the engine's folded status snapshot:
 * provider-reported token counters (`usage`) alongside the current
 * context size (`contextTokens` / `maxContextTokens`). The adapter turns
 * each one into an ACP `usage_update` so a client can paint a live
 * context meter mid-turn, and reports the turn total once more on the
 * `PromptResponse`.
 */
describe('AcpServer session/prompt usage reporting', () => {
  const tokens = (
    inputOther: number,
    output: number,
    inputCacheRead = 0,
    inputCacheCreation = 0,
  ) => ({ inputOther, output, inputCacheRead, inputCacheCreation });

  const statusEvent = (
    sessionId: string,
    overrides: Record<string, unknown>,
  ): Event =>
    ({
      type: 'agent.status.updated',
      sessionId,
      agentId: 'main',
      model: 'kimi-code/k3',
      ...overrides,
    }) as unknown as Event;

  const usageUpdates = (collecting: CollectingClient) =>
    collecting.updates.filter(
      (n) => (n.update as { sessionUpdate?: string }).sessionUpdate === 'usage_update',
    );

  async function runTurn(script: readonly Event[], sessionId: string) {
    const { session } = makeScriptedSession(sessionId, script);
    const harness = {
      auth: { status: async () => AUTHED_STATUS },
      createSession: async () => session,
    } as unknown as KimiHarness;

    const { agentStream, clientStream } = makeInMemoryStreamPair();
    new AgentSideConnection((c) => new AcpServer(harness, c), agentStream);
    const collecting = new CollectingClient();
    const client = new ClientSideConnection(() => collecting, clientStream);

    await client.newSession({ cwd: '/tmp/x', mcpServers: [] });
    const response = await client.prompt({ sessionId, prompt: [textBlock('hi')] });
    // Give the agent side a tick to flush queued sessionUpdate writes.
    await new Promise((resolve) => setTimeout(resolve, 20));
    return { response, collecting };
  }

  it('streams a usage_update per model call and returns the turn total on the response', async () => {
    const sessionId = 'sess-usage-A';
    const { response, collecting } = await runTurn(
      [
        statusEvent(sessionId, {
          contextTokens: 12_000,
          maxContextTokens: 1_048_576,
          usage: {
            currentTurn: tokens(2_171, 70, 9_759),
            total: tokens(2_171, 70, 9_759),
            byModel: { 'kimi-code/k3': tokens(2_171, 70, 9_759) },
          },
        }),
        { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'hi' } as Event,
        statusEvent(sessionId, {
          contextTokens: 21_185,
          maxContextTokens: 1_048_576,
          usage: {
            currentTurn: tokens(3_500, 120, 18_944),
            total: tokens(3_500, 120, 18_944),
            byModel: { 'kimi-code/k3': tokens(3_500, 120, 18_944) },
          },
        }),
        { type: 'turn.ended', sessionId, agentId: 'main', turnId: 1, reason: 'completed' } as Event,
      ],
      sessionId,
    );

    const updates = usageUpdates(collecting);
    expect(updates).toHaveLength(2);
    expect(updates[0]?.update).toMatchObject({
      sessionUpdate: 'usage_update',
      used: 12_000,
      size: 1_048_576,
      _meta: {
        model: 'kimi-code/k3',
        usage: {
          inputTokens: 2_171,
          outputTokens: 70,
          cachedReadTokens: 9_759,
          cachedWriteTokens: 0,
          totalTokens: 12_000,
        },
      },
    });
    expect(updates[1]?.update).toMatchObject({
      sessionUpdate: 'usage_update',
      used: 21_185,
      size: 1_048_576,
      _meta: {
        usage: { inputTokens: 3_500, outputTokens: 120, cachedReadTokens: 18_944 },
        sessionUsage: { inputTokens: 3_500, outputTokens: 120, cachedReadTokens: 18_944 },
      },
    });

    // The response carries the turn's final running total, not a
    // session-cumulative figure.
    expect(response.usage).toEqual({
      inputTokens: 3_500,
      outputTokens: 120,
      cachedReadTokens: 18_944,
      cachedWriteTokens: 0,
      totalTokens: 22_564,
    });
  });

  it('separates the per-turn running total from the session-cumulative total', async () => {
    const sessionId = 'sess-usage-B';
    const { response, collecting } = await runTurn(
      [
        statusEvent(sessionId, {
          contextTokens: 30_000,
          maxContextTokens: 262_144,
          usage: {
            currentTurn: tokens(500, 40),
            total: tokens(9_000, 640, 20_000, 1_000),
          },
        }),
        { type: 'turn.ended', sessionId, agentId: 'main', turnId: 2, reason: 'completed' } as Event,
      ],
      sessionId,
    );

    expect(usageUpdates(collecting)[0]?.update).toMatchObject({
      _meta: {
        usage: { totalTokens: 540 },
        sessionUsage: { totalTokens: 30_640 },
      },
    });
    expect(response.usage).toMatchObject({ totalTokens: 540 });
  });

  it('drops duplicate status slices, subagent status, and windowless snapshots', async () => {
    const sessionId = 'sess-usage-C';
    const snapshot = {
      contextTokens: 5_000,
      maxContextTokens: 131_072,
      usage: { currentTurn: tokens(100, 10), total: tokens(100, 10) },
    };
    const { response, collecting } = await runTurn(
      [
        statusEvent(sessionId, snapshot),
        // Same folded snapshot re-published on a different status slice
        // (e.g. a plan-mode toggle) — must not repaint the meter.
        statusEvent(sessionId, snapshot),
        // A subagent's own status must not overwrite the parent meter.
        statusEvent(sessionId, {
          agentId: 'sub-1',
          contextTokens: 999,
          maxContextTokens: 131_072,
          usage: { currentTurn: tokens(1, 1), total: tokens(1, 1) },
        }),
        // No model bound yet: ACP `size` has no "unknown" encoding, so a
        // snapshot without a context limit produces no update at all.
        statusEvent(sessionId, {
          contextTokens: 7_000,
          usage: { currentTurn: tokens(200, 20), total: tokens(200, 20) },
        }),
        { type: 'turn.ended', sessionId, agentId: 'main', turnId: 3, reason: 'completed' } as Event,
      ],
      sessionId,
    );

    const updates = usageUpdates(collecting);
    expect(updates).toHaveLength(1);
    expect(updates[0]?.update).toMatchObject({ used: 5_000, size: 131_072 });
    // The windowless snapshot still advances the turn total even though
    // it produced no meter update.
    expect(response.usage).toMatchObject({ totalTokens: 220 });
  });

  it('omits usage from the response when no status event carried a turn total', async () => {
    const sessionId = 'sess-usage-D';
    const { response, collecting } = await runTurn(
      [
        { type: 'assistant.delta', sessionId, agentId: 'main', turnId: 1, delta: 'hi' } as Event,
        { type: 'turn.ended', sessionId, agentId: 'main', turnId: 1, reason: 'completed' } as Event,
      ],
      sessionId,
    );

    expect(usageUpdates(collecting)).toHaveLength(0);
    expect(response.usage ?? undefined).toBeUndefined();
  });
});
