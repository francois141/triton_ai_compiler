/**
 * Tools for the Pi agent and the workspace read-only jail.
 *
 * Exposes:
 *   - `makeSubmitPtxTool`  -- the custom submit_ptx tool (compile/verify/benchmark)
 *   - `READ_ONLY_TOOL_NAMES` -- built-in tools the agent is allowed to enable
 *   - `WORKSPACE_DIR`      -- the folder the read-only tools are confined to
 *   - `makeWorkspaceJail`  -- an inline extension that blocks any read-only tool
 *                             call whose path escapes WORKSPACE_DIR
 *
 * Skills live under WORKSPACE_DIR so the model can discover and read them while
 * being unable to read anything else on the machine.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import { evaluateCandidate, TritonApiError } from "./triton_api.js";
import { GRAY, GREEN, RED, RESET } from "./tui.js";

// --- Workspace ---

/**
 * The single folder the agent may read from. Created if missing so the jail has
 * a stable realpath to compare against. Store skills here (e.g. as
 * `agent_workspace/<skill>/SKILL.md`).
 */
export const WORKSPACE_DIR = (() => {
  const dir = fileURLToPath(new URL("./agent_workspace", import.meta.url));
  fs.mkdirSync(dir, { recursive: true });
  return fs.realpathSync(dir);
})();

/**
 * Skill directories, each containing a SKILL.md. They live INSIDE WORKSPACE_DIR
 * (copied there from their source, e.g. the `ptx_skills` submodule) so the jail
 * already permits reading them. They are advertised to the model via the
 * loader's `additionalSkillPaths` (see agent.js) -- copying alone makes them
 * readable but not discovered, since Pi does not auto-scan the workspace root.
 */
export const SKILL_DIRS = [path.join(WORKSPACE_DIR, "ptx_skill")];

/**
 * The built-in, read-only tools the agent is allowed to use. These only ever
 * read; combined with the jail below they cannot touch anything outside
 * WORKSPACE_DIR. All four take an optional `path` argument (ls/grep/find default
 * it to the cwd, which is WORKSPACE_DIR).
 */
export const READ_ONLY_TOOL_NAMES = ["read", "ls", "grep", "find"];

const READ_ONLY_TOOL_SET = new Set(READ_ONLY_TOOL_NAMES);

/** True if `rawPath` (resolved against the workspace) stays inside it, symlinks included. */
function insideWorkspace(rawPath) {
  const abs = path.resolve(WORKSPACE_DIR, rawPath);
  // Realpath the nearest existing ancestor so a symlink that points outside is
  // caught even when the leaf does not exist yet.
  let probe = abs;
  while (!fs.existsSync(probe) && probe !== path.dirname(probe)) {
    probe = path.dirname(probe);
  }
  let real;
  try {
    real = fs.realpathSync(probe);
  } catch {
    return false;
  }
  return real === WORKSPACE_DIR || real.startsWith(WORKSPACE_DIR + path.sep);
}

/**
 * Inline extension that confines the read-only tools to WORKSPACE_DIR. The
 * `tool_call` event fires before every tool executes and can block it; a thrown
 * error also blocks (fail-safe). submit_ptx has no path and is unaffected.
 */
export function makeWorkspaceJail() {
  return {
    name: "workspace-jail",
    factory: (pi) => {
      pi.on("tool_call", (event) => {
        if (!READ_ONLY_TOOL_SET.has(event.toolName)) return;
        // read uses `path`; older arg shape also accepts `file_path`.
        const raw = event.input?.path ?? event.input?.file_path;
        // Omitted path means "the cwd" (WORKSPACE_DIR) -- allowed.
        if (raw === undefined || raw === null || raw === "") return;
        if (!insideWorkspace(String(raw))) {
          console.log(
            `${RED}[denied]${RESET} ${GRAY}${event.toolName} tried to access ` +
              `${raw} outside the workspace${RESET}`,
          );
          return {
            block: true,
            reason:
              `Path outside the agent workspace is not allowed: ${raw}. ` +
              `Read-only tools may only access ${WORKSPACE_DIR}.`,
          };
        }
      });
    },
  };
}

// --- submit_ptx tool ---

/** Build the single submit_ptx tool bound to a kernel id. */
export function makeSubmitPtxTool(kernelId, tracker) {
  return defineTool({
    name: "submit_ptx",
    label: "Submit PTX",
    description:
      "Compile, verify, and benchmark a PTX candidate for the kernel. " +
      "Returns whether it compiled, whether it is correct, its p50 latency, " +
      "and its speedup versus the Triton baseline.",
    parameters: Type.Object({
      ptx: Type.String({
        description: "The full PTX module source for the candidate.",
      }),
      num_threads_x: Type.Integer({
        description: "Number of threads per block along x (required).",
      }),
      num_threads_y: Type.Optional(
        Type.Integer({ description: "Number of threads per block along y (optional)." }),
      ),
      num_threads_z: Type.Optional(
        Type.Integer({ description: "Number of threads per block along z (optional)." }),
      ),
    }),
    execute: async (_toolCallId, params) => {
      // The "[call]" marker is rendered from the tool_execution_start event so
      // streaming stays in one place; here we only evaluate and report results.
      let result;
      try {
        result = await evaluateCandidate(kernelId, params);
      } catch (err) {
        // Surface environment/transport errors back to the model as text.
        const message =
          err instanceof TritonApiError
            ? `Evaluation failed (HTTP ${err.status}): ${err.message}`
            : `Evaluation failed: ${err.message}`;
        console.log(`${RED}[error]${RESET} ${GRAY}${message}${RESET}`);
        return {
          content: [{ type: "text", text: JSON.stringify({ error: message }, null, 2) }],
          details: { error: message },
        };
      }

      if (result.passed && result.speedup_vs_triton > tracker.bestSpeedup) {
        tracker.bestSpeedup = result.speedup_vs_triton;
      }

      const tag = result.passed ? `${GREEN}[ok]${RESET}` : `${RED}[fail]${RESET}`;
      const text = JSON.stringify(result, null, 2);
      console.log(`${tag} ${GRAY}${text}${RESET}`);

      return {
        content: [{ type: "text", text }],
        details: {},
      };
    },
  });
}
