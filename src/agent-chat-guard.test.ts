// agent-chat-guard.test.ts — plain-assert tests, run with:  npm test
// Covers: chain provenance parsing, human-rooted vs spontaneous permissions,
// @-back + hop stamping, mention-strip at the cap, chain termination, and the
// timeout/interrupt status edits (via a fake Slack client).
import * as assert from 'assert';
import {
  MAX_HOPS,
  READONLY_ALLOWED_TOOLS,
  BOT_TURN_HARD_EXCLUSIONS,
  inspectInbound,
  applyReadOnlyForBotTurn,
  stampOutgoing,
  ChainInfo,
} from './agent-chat-guard';
import { SlackHandler, TIMEOUT_STATUS, INTERRUPTED_STATUS } from './slack-handler';

const STRIP_RE = /<@[A-Z0-9]+>|@(gilbert|kathryne|charizard|awesome-ash|sam|sadie)\b/gi;
const AGENT_RE = /<@(U0B9C278VPW|U0BA8P14L5N|U0B9CFA6KFY|U0B9X82Q5FX|U0B9Y47N1EH|U0B9L2MEKUP)>/i;
const ROOT = '1718000000.000100';

let passed = 0;
function test(name: string, fn: () => void | Promise<void>) {
  const result = fn();
  const done = () => { passed++; console.log(`PASS  ${name}`); };
  if (result instanceof Promise) return result.then(done);
  done();
}

const tests: Array<Promise<void> | void> = [];

// --- parsing ---------------------------------------------------------------

tests.push(test('inspectInbound parses a provenance tag', () => {
  const c = inspectInbound('B123', `do the thing\n\n_[chain:${ROOT} hop 2/${MAX_HOPS}]_`);
  assert.strictEqual(c.isFromBot, true);
  assert.strictEqual(c.hop, 2);
  assert.strictEqual(c.rootTs, ROOT);
  assert.strictEqual(c.atLimit, false);
  assert.strictEqual(c.humanRooted, false); // never from the tag — handler-verified only
}));

tests.push(test('inspectInbound parses a legacy (unrooted) hop tag', () => {
  const c = inspectInbound('B123', 'hello\n\n_[hop 3/4]_');
  assert.strictEqual(c.hop, 3);
  assert.strictEqual(c.rootTs, undefined);
}));

tests.push(test('hop just below the cap flags atLimit', () => {
  const c = inspectInbound('B123', `x _[chain:${ROOT} hop ${MAX_HOPS - 1}/${MAX_HOPS}]_`);
  assert.strictEqual(c.atLimit, true);
}));

// --- permissions -----------------------------------------------------------

const baseOptions = () => ({ permissionMode: 'bypassPermissions', cwd: '/repo' });

tests.push(test('human-addressed turn keeps full tools untouched', () => {
  const chain: ChainInfo = { isFromBot: false, hop: 0, atLimit: false, humanRooted: false };
  const opts = applyReadOnlyForBotTurn(baseOptions(), chain);
  assert.deepStrictEqual(opts, baseOptions());
}));

tests.push(test('spontaneous bot chain stays read-only', () => {
  const chain: ChainInfo = { isFromBot: true, hop: 1, atLimit: false, humanRooted: false };
  const opts = applyReadOnlyForBotTurn(baseOptions(), chain);
  assert.strictEqual(opts.permissionMode, 'default');
  assert.deepStrictEqual(opts.allowedTools, READONLY_ALLOWED_TOOLS);
  assert.ok(opts.disallowedTools.includes('Bash'));
  assert.ok(opts.disallowedTools.includes('Write'));
}));

tests.push(test('human-rooted chain may do work, minus hard exclusions', () => {
  const chain: ChainInfo = { isFromBot: true, hop: 1, atLimit: false, humanRooted: true, rootTs: ROOT };
  const opts = applyReadOnlyForBotTurn(baseOptions(), chain);
  assert.strictEqual(opts.permissionMode, 'bypassPermissions'); // full tools kept
  assert.strictEqual(opts.allowedTools, undefined);             // no read-only allowlist
  assert.ok(!opts.disallowedTools.includes('Bash'));            // bash allowed in general...
  for (const rule of BOT_TURN_HARD_EXCLUSIONS) {
    assert.ok(opts.disallowedTools.includes(rule));             // ...but not start/stop/merge
  }
  assert.ok(opts.disallowedTools.includes('Bash(gh pr merge:*)'));
  assert.ok(opts.disallowedTools.includes('Bash(aws ec2 start-instances:*)'));
}));

// --- stamping --------------------------------------------------------------

tests.push(test('human turn that @mentions an agent roots a chain at hop 1', () => {
  const chain: ChainInfo = { isFromBot: false, hop: 0, atLimit: false, humanRooted: false };
  const out = stampOutgoing('<@U0BA8P14L5N> please check the calib file', chain,
    { stripRe: STRIP_RE, agentRe: AGENT_RE, rootTs: ROOT });
  assert.ok(out.includes(`[chain:${ROOT} hop 1/${MAX_HOPS}]`));
}));

tests.push(test('human turn without an agent mention is not stamped', () => {
  const chain: ChainInfo = { isFromBot: false, hop: 0, atLimit: false, humanRooted: false };
  const out = stampOutgoing('all done, results in data/', chain,
    { stripRe: STRIP_RE, agentRe: AGENT_RE, rootTs: ROOT });
  assert.strictEqual(out, 'all done, results in data/');
}));

tests.push(test('bot turn @-backs the initiator and advances the hop', () => {
  const chain: ChainInfo = {
    isFromBot: true, hop: 2, atLimit: false, humanRooted: true,
    rootTs: ROOT, initiatorId: 'U0BA8P14L5N',
  };
  const out = stampOutgoing('done — synced to s3', chain, { stripRe: STRIP_RE, agentRe: AGENT_RE });
  assert.ok(out.startsWith('<@U0BA8P14L5N> '));
  assert.ok(out.includes(`[chain:${ROOT} hop 3/${MAX_HOPS}]`));
}));

tests.push(test('at the cap: mentions stripped, chain ended, no @-back', () => {
  const chain: ChainInfo = {
    isFromBot: true, hop: MAX_HOPS - 1, atLimit: true, humanRooted: false,
    rootTs: ROOT, initiatorId: 'U0BA8P14L5N',
  };
  const out = stampOutgoing('ask <@U0B9C278VPW> or @sam about it', chain,
    { stripRe: STRIP_RE, agentRe: AGENT_RE });
  assert.ok(!/<@U0B9C278VPW>/.test(out));
  assert.ok(!/@sam\b/.test(out));
  assert.ok(!out.startsWith('<@U0BA8P14L5N>'));
  assert.ok(out.includes(`hop ${MAX_HOPS}/${MAX_HOPS}] — chain ended`));
}));

tests.push(test('a human-rooted chain terminates within the hop cap', () => {
  // Simulate the full ping-pong: each bot's stamped output is the next bot's input.
  let text = stampOutgoing('<@U0B9C278VPW> kick it off', // human turn roots the chain
    { isFromBot: false, hop: 0, atLimit: false, humanRooted: false },
    { stripRe: STRIP_RE, agentRe: AGENT_RE, rootTs: ROOT });
  let turns = 0;
  for (; turns < 20; turns++) {
    const chain = inspectInbound('B_PEER', text);
    if (chain.hop >= MAX_HOPS) break; // the handler's hard stop: no reply at all
    chain.initiatorId = 'U0B9Y47N1EH';
    text = stampOutgoing('still discussing <@U0B9C278VPW>', chain, { stripRe: STRIP_RE, agentRe: AGENT_RE });
  }
  assert.ok(turns <= MAX_HOPS, `chain ran ${turns} bot turns, cap is ${MAX_HOPS}`);
  assert.ok(text.includes('chain ended'));
}));

// --- timeout / interrupt status edits (fake Slack client) -------------------

function fakeHandler() {
  const updates: any[] = [];
  const app = {
    client: {
      chat: { update: async (args: any) => { updates.push(args); return { ok: true }; } },
      reactions: {
        add: async () => ({ ok: true }),
        remove: async () => ({ ok: true }),
      },
    },
  };
  const handler = new SlackHandler(app as any, {} as any, {} as any);
  return { handler: handler as any, updates };
}

tests.push(test('timeout edits the status message to :stopwatch:', async () => {
  const { handler, updates } = fakeHandler();
  await handler.finishAbortedTurn('sess-1', '111.222', 'C0CHAN', true);
  assert.strictEqual(updates.length, 1);
  assert.strictEqual(updates[0].text, TIMEOUT_STATUS);
  assert.strictEqual(updates[0].ts, '111.222');
}));

tests.push(test('plain abort (not timeout) stays a cancellation', async () => {
  const { handler, updates } = fakeHandler();
  await handler.finishAbortedTurn('sess-1', '111.222', 'C0CHAN', false);
  assert.strictEqual(updates[0].text, '⏹️ *Cancelled*');
}));

tests.push(test('SIGTERM edit marks in-flight status as interrupted', async () => {
  const { handler, updates } = fakeHandler();
  handler.activeControllers.set('sess-1', new AbortController());
  handler.statusMessages.set('sess-1', { channel: 'C0CHAN', ts: '333.444' });
  const realExit = process.exit;
  let exited = false;
  (process as any).exit = () => { exited = true; };
  try {
    await handler.handleShutdownSignal('SIGTERM');
  } finally {
    (process as any).exit = realExit;
  }
  assert.ok(exited);
  assert.strictEqual(updates.length, 1);
  assert.strictEqual(updates[0].text, INTERRUPTED_STATUS);
}));

Promise.all(tests).then(() => {
  console.log(`\n${passed} tests passed`);
}).catch((err) => {
  console.error('FAIL', err);
  process.exit(1);
});
