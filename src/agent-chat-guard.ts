// agent-chat-guard.ts
// Bounded bot-to-bot conversation guards for the CalibrateRL Slack listener.
//
// Three guarantees:
//   1. PROVENANCE GATE: every chain message carries a hidden
//      [chain:<root_ts> hop N/MAX] tag. If the root message (looked up by the
//      handler, fail-safe) was authored by a real HUMAN, the chain is
//      "human-rooted" and bot-initiated turns MAY do work (files, commands,
//      propose-pr). No verified human root -> the turn is READ-ONLY: no
//      write/exec tools at all. Hard exclusions apply to EVERY bot-initiated
//      turn regardless of rooting: no instance start/stop, no merging.
//   2. HOP LIMIT: at the limit the replying bot is stripped of the ability to
//      tag another bot, so the chain terminates. A human @mention resets it.
//   3. @-BACK: bot-turn replies open with the initiating agent's @mention so
//      the answer lands back with whoever asked.
//
// This is intentionally small and self-contained so it can be applied identically
// on gilbert / kathryne / charizard and removed cleanly.

export const MAX_HOPS = 5; // bot<->bot exchanges before the chain is force-ended

// Hidden markers appended to chain messages. Humans don't type these; only bots
// emit them, so their presence is how we know a message came from "the chain".
//   provenance form: [chain:<root_ts> hop N/5]  (root_ts = Slack ts of the
//                    message that rooted the chain)
//   legacy form:     [hop N/4]  (pre-provenance; treated as an UNROOTED chain,
//                    so it stays read-only)
const CHAIN_RE = /\[chain:(\d+\.\d+)\s+hop\s+(\d+)\/(\d+)\]/i;
const HOP_RE = /\[hop\s+(\d+)\/(\d+)\]/i;

// Read-only MCP/tool allowlist for unrooted bot-to-bot turns. Adjust to your
// server names. These let the bot read the repo + read Slack, but NOT write
// files, run bash that mutates, or push. (Claude Code's built-in Read/Grep/Glob
// are safe; we simply do NOT include Write/Edit/Bash in the allowlist, and we
// keep permissionMode strict so nothing unlisted runs.)
export const READONLY_ALLOWED_TOOLS = [
  'Read', 'Grep', 'Glob', 'WebSearch', 'WebFetch',
  'mcp__slack__slack_get_channel_history',
  'mcp__slack__slack_list_channels',
  'mcp__slack__slack_get_user_profile',
  'mcp__slack__slack_get_users',
];

// Denied on EVERY bot-initiated turn, even human-rooted ones: starting/stopping
// instances and merging stay with humans (and with directly human-addressed
// turns). Deny rules win over allows and over bypassPermissions.
export const BOT_TURN_HARD_EXCLUSIONS = [
  'Bash(aws ec2 start-instances:*)',
  'Bash(aws ec2 stop-instances:*)',
  'Bash(aws ec2 terminate-instances:*)',
  'Bash(start_box.sh:*)',
  'Bash(/usr/local/bin/start_box.sh:*)',
  'Bash(shutdown:*)',
  'Bash(sudo shutdown:*)',
  'Bash(gh pr merge:*)',
  'Bash(git merge:*)',
];

export interface ChainInfo {
  isFromBot: boolean;   // inbound message was authored by another bot
  hop: number;          // current hop number parsed from inbound (0 if none)
  atLimit: boolean;     // true if this turn is the last allowed hop
  rootTs?: string;      // provenance: ts of the chain's root message, if tagged
  humanRooted: boolean; // set by the HANDLER after verifying the root message's
                        // author is a real human — never trusted from the tag
  initiatorId?: string; // Slack user id of the agent that @mentioned us (@-back)
}

// Inspect an inbound Slack event. `botId` is event.bot_id (set by Slack when the
// author is a bot/app). `text` is the message body.
export function inspectInbound(botId: string | undefined, text: string | undefined): ChainInfo {
  const isFromBot = !!botId;
  let hop = 0;
  let rootTs: string | undefined;
  if (text) {
    const c = text.match(CHAIN_RE);
    if (c) {
      rootTs = c[1];
      hop = parseInt(c[2], 10) || 0;
    } else {
      const m = text.match(HOP_RE);
      if (m) hop = parseInt(m[1], 10) || 0;
    }
  }
  const nextHop = hop + 1;
  return {
    isFromBot,
    hop,
    atLimit: nextHop >= MAX_HOPS,
    rootTs,
    humanRooted: false,
  };
}

// Decide the tool options for this turn.
//   human-addressed          => full tools (caller's default), untouched
//   bot turn, human-rooted   => full tools MINUS the hard exclusions
//   bot turn, unrooted       => read-only allowlist, strict permission mode
export function applyReadOnlyForBotTurn(options: any, chain: ChainInfo): any {
  if (!chain.isFromBot) return options; // human-addressed: leave full tools intact
  if (chain.humanRooted) {
    return {
      ...options,
      disallowedTools: [
        ...(options.disallowedTools || []),
        ...BOT_TURN_HARD_EXCLUSIONS,
      ],
    };
  }
  return {
    ...options,
    permissionMode: 'default',            // strict: only listed tools may run
    allowedTools: READONLY_ALLOWED_TOOLS, // no Write/Edit/Bash/git
    disallowedTools: ['Write', 'Edit', 'Bash', 'NotebookEdit', ...BOT_TURN_HARD_EXCLUSIONS],
  };
}

export interface StampOpts {
  stripRe: RegExp;   // matches ANY mention (global) — drives the strip at the cap
  agentRe?: RegExp;  // matches a known agent-bot user mention (NON-global) —
                     // detects a chain start on human turns
  rootTs?: string;   // ts of this turn's inbound message — becomes the chain
                     // root when a human turn's reply @mentions another agent
}

// Post-process the bot's outgoing text before it goes to Slack.
//   human turn whose reply @mentions an agent -> stamp [chain:<root> hop 1/N]
//     (that mention starts a chain rooted at the human's message)
//   bot turn -> @-back prefix + next hop counter, preserving the chain root;
//     at the limit, strip any mentions so the chain can't continue
export function stampOutgoing(text: string, chain: ChainInfo, opts: StampOpts): string {
  if (!chain.isFromBot) {
    if (opts.rootTs && opts.agentRe && opts.agentRe.test(text)) {
      return text + `\n\n_[chain:${opts.rootTs} hop 1/${MAX_HOPS}]_`;
    }
    return text; // plain human turn: don't stamp
  }
  const nextHop = chain.hop + 1;
  const tag = chain.rootTs
    ? `[chain:${chain.rootTs} hop ${nextHop}/${MAX_HOPS}]`
    : `[hop ${nextHop}/${MAX_HOPS}]`;
  let out = text;
  if (chain.atLimit) {
    // Force-end: remove mentions so no other bot is triggered. (This also
    // removes any would-be @-back — the chain is over, nobody should re-enter.)
    out = out.replace(opts.stripRe, '(chat limit reached — ending here)');
    out += `\n\n_${tag} — chain ended_`;
  } else {
    // @-back: open with the initiating agent's mention so the reply is
    // addressed to whoever asked.
    if (chain.initiatorId && !out.startsWith(`<@${chain.initiatorId}>`)) {
      out = `<@${chain.initiatorId}> ` + out;
    }
    out += `\n\n_${tag}_`;
  }
  return out;
}
