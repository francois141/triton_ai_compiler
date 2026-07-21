/** ANSI colors and the TUI-style streaming renderer shared across the agent. */

export const RESET = "\x1b[0m";
export const BLUE = "\x1b[34m";
export const GREEN = "\x1b[32m";
export const RED = "\x1b[31m";
export const GRAY = "\x1b[90m";
export const CYAN = "\x1b[36m";
export const MAGENTA = "\x1b[35m";
export const YELLOW = "\x1b[33m";

/**
 * Attach the TUI-style streaming renderer to a session. Shared by the lead
 * agent (agent.js) and the subagents (subagent.js) so both display their
 * reasoning and tool calls the same way, without duplicating the (fairly large)
 * event handler.
 *
 * It renders reasoning (grey), answer text (cyan), tool calls, per-turn
 * stop-reason info, and compaction notices, with clean transitions between the
 * streamed modes. Pass `label` to prefix every section header (e.g.
 * "[subagent 0]") so nested subagent output is visually distinct from the lead
 * agent's; with no label the output is byte-for-byte what the lead agent
 * printed before.
 *
 * @param {import("@earendil-works/pi-coding-agent").AgentSession} session
 * @param {object} [opts]
 * @param {string} [opts.label]  Header prefix, e.g. "[subagent 0]". Empty for
 *   the lead agent (no prefix).
 * @returns {{ getLastStopReason: () => string | null }} Accessor for the last
 *   turn's stop reason, which the caller may inspect after the run finishes.
 */
export function renderSessionOutput(session, { label = "" } = {}) {
  // Grey label shown on every header line; empty (no leading space) for the lead.
  const tag = label ? `${GRAY}${label}${RESET} ` : "";
  // Plain label for the streamed thinking/text prefixes (already inside a color).
  const inlineTag = label ? `${label} ` : "";

  let streamMode = null; // "text" | "thinking" | null
  let lastStopReason = null;

  session.subscribe((event) => {
    switch (event.type) {
      case "message_update": {
        const ev = event.assistantMessageEvent;
        if (ev.type === "thinking_start" || ev.type === "thinking_delta") {
          if (streamMode !== "thinking") {
            process.stdout.write(`\n${GRAY}${inlineTag}[thinking] `);
            streamMode = "thinking";
          }
          if (ev.type === "thinking_delta") process.stdout.write(ev.delta);
        } else if (ev.type === "text_start" || ev.type === "text_delta") {
          if (streamMode !== "text") {
            process.stdout.write(`\n${CYAN}${inlineTag}`);
            streamMode = "text";
          }
          if (ev.type === "text_delta") process.stdout.write(ev.delta);
        } else if (ev.type === "text_end" || ev.type === "thinking_end") {
          process.stdout.write(RESET);
          streamMode = null;
        }
        break;
      }
      case "tool_execution_start": {
        process.stdout.write(RESET);
        const args = event.args || {};
        let summary;
        if (event.toolName === "submit_ptx") {
          const ptxLen = typeof args.ptx === "string" ? `${args.ptx.length} chars` : "?";
          summary = `submit_ptx(ptx: ${ptxLen}, num_threads_x: ${args.num_threads_x})`;
        } else if (event.toolName === "spawn_subagent") {
          const instr = typeof args.instructions === "string" ? args.instructions : "";
          const preview = instr.replace(/\s+/g, " ").slice(0, 80);
          summary = `spawn_subagent(${preview}${instr.length > 80 ? "…" : ""})`;
        } else {
          // Read-only tools (read/ls/grep/find): show the most useful arg.
          const detail = args.pattern ?? args.path ?? "";
          summary = `${event.toolName}(${detail})`;
        }
        console.log(`\n${tag}${BLUE}[call]${RESET} ${GRAY}${summary}${RESET}`);
        streamMode = null;
        break;
      }
      case "turn_end": {
        // Report why each turn ended, so a run that stops without submitting is
        // diagnosable. stopReason "length" means the model hit its output-token
        // cap (maxTokens) mid-generation, likely before calling submit_ptx.
        const msg = event.message || {};
        const usage = msg.usage;
        lastStopReason = msg.stopReason ?? null;
        process.stdout.write(RESET);
        const usageStr = usage
          ? `${usage.output} out` +
            (usage.reasoning != null ? ` (${usage.reasoning} reasoning)` : "") +
            `, ${usage.input} in`
          : "usage n/a";
        console.log(
          `\n${tag}${GRAY}[info] stop=${lastStopReason} | ${usageStr}${RESET}`,
        );
        if (lastStopReason === "length") {
          console.log(
            `${tag}${RED}[warning]${RESET} ${GRAY}Turn hit the model output-token limit ` +
              `(maxTokens) mid-generation, so it may have stopped before calling ` +
              `submit_ptx. Raise the model's maxTokens or lower thinkingLevel.${RESET}`,
          );
        }
        streamMode = null;
        break;
      }
      case "compaction_start": {
        // Pi auto-compacts when context passes contextWindow - reserveTokens
        // (reason "threshold") or after a context-overflow response (reason
        // "overflow"); "manual" is an explicit session.compact(). It summarizes
        // older turns and keeps the recent ~keepRecentTokens verbatim.
        process.stdout.write(RESET);
        console.log(
          `\n${tag}${YELLOW}[compact]${RESET} ${GRAY}start (${event.reason})...${RESET}`,
        );
        streamMode = null;
        break;
      }
      case "compaction_end": {
        process.stdout.write(RESET);
        const { reason, result, aborted, willRetry, errorMessage } = event;
        if (errorMessage) {
          console.log(
            `${tag}${RED}[compact]${RESET} ${GRAY}failed (${reason}): ${errorMessage}${RESET}`,
          );
        } else if (aborted) {
          console.log(`${tag}${YELLOW}[compact]${RESET} ${GRAY}aborted (${reason})${RESET}`);
        } else {
          const before = result?.tokensBefore;
          const after = result?.estimatedTokensAfter;
          const delta =
            before != null && after != null
              ? `${before} -> ~${after} tokens`
              : "tokens n/a";
          const retryNote = willRetry ? ", retrying turn" : "";
          console.log(
            `${tag}${YELLOW}[compact]${RESET} ${GRAY}done (${reason}): ${delta}${retryNote}${RESET}`,
          );
        }
        streamMode = null;
        break;
      }
    }
  });

  return { getLastStopReason: () => lastStopReason };
}
