/**
 * Tools for the Pi agent and the workspace read-only jail.
 *
 * Exposes:
 *   - `makeSubmitPtxTool`  -- the custom submit_ptx tool (compile/verify/benchmark),
 *                             which also persists new record kernels to BEST_KERNEL_DIR
 *   - `BEST_KERNEL_DIR`    -- where the top PTX per kernel is accumulated
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
import { GRAY, GREEN, MAGENTA, RED, RESET } from "./tui.js";

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

/** Filename (in WORKSPACE_DIR) where the kernel's Triton source is saved. */
export const TRITON_SOURCE_FILE = "triton_source.py";

/**
 * Save the kernel's original Triton source into the workspace so the agent and
 * its subagents (which do not share the parent's context) can read it. Returns
 * the filename. Overwrites any previous run's copy.
 */
export function saveTritonSource(source) {
  fs.writeFileSync(path.join(WORKSPACE_DIR, TRITON_SOURCE_FILE), source);
  return TRITON_SOURCE_FILE;
}

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

// --- Best-kernel persistence ---

/**
 * Directory where the top PTX per kernel is accumulated, indexed by kernel id:
 *   best_kernel/<kernelId>/<speedup>-<sessionId>.ptx   (speedup as 5 decimals)
 * A candidate is saved only when its speedup strictly beats every .ptx already
 * saved for that kernel -- an append-only history of record-beaters. The
 * session id in the filename links each result back to its saved session.
 */
export const BEST_KERNEL_DIR = fileURLToPath(new URL("./best_kernel", import.meta.url));

/** Highest speedup encoded in the filenames already saved for a kernel, or 0. */
function bestSavedSpeedup(kernelDir) {
  let best = 0;
  let entries;
  try {
    entries = fs.readdirSync(kernelDir);
  } catch {
    return best; // dir does not exist yet -> nothing saved
  }
  for (const name of entries) {
    if (!name.endsWith(".ptx")) continue;
    // Filename is "<speedup>-<sessionId>.ptx"; the speedup uses a dot and the
    // session id owns the first hyphen, so parseFloat of the leading token is it.
    const speedup = parseFloat(name);
    if (Number.isFinite(speedup) && speedup > best) best = speedup;
  }
  return best;
}

/**
 * Persist `ptx` as a new best for `kernelId` iff its (5-decimal) speedup beats
 * every saved candidate. Returns the written path, or null if not saved.
 * Best-effort: filesystem errors are logged but never break the tool.
 */
function saveBestKernel(kernelId, ptx, speedup, sessionId) {
  const kernelDir = path.join(BEST_KERNEL_DIR, kernelId);
  const rounded = Number(speedup.toFixed(5));
  if (rounded <= bestSavedSpeedup(kernelDir)) return null;
  const file = path.join(kernelDir, `${speedup.toFixed(5)}-${sessionId}.ptx`);
  try {
    fs.mkdirSync(kernelDir, { recursive: true });
    fs.writeFileSync(file, ptx);
    return file;
  } catch (err) {
    console.log(`${RED}[warn]${RESET} ${GRAY}Could not save best kernel: ${err.message}${RESET}`);
    return null;
  }
}

// --- submit_ptx tool ---

/**
 * Return a copy of `result` without the server-echoed PTX (`payload.ptx`). The
 * PTX is already in context from the tool call, so echoing it back in the
 * response just wastes tokens.
 */
function stripEchoedPtx(result) {
  if (!result || typeof result !== "object" || !result.payload) return result;
  const { ptx: _ptx, ...payloadRest } = result.payload;
  return { ...result, payload: payloadRest };
}

/**
 * Write the submitted PTX (`ptxFile`) and the trimmed result (`resultFile`) to
 * the workspace so the agent can read them back later. Returns a note describing
 * where they landed (or the failure). Best-effort: never throws.
 */
function saveKernelArtifacts(ptxFile, resultFile, ptx, trimmedResult) {
  try {
    fs.writeFileSync(path.join(WORKSPACE_DIR, ptxFile), ptx);
    fs.writeFileSync(
      path.join(WORKSPACE_DIR, resultFile),
      JSON.stringify(trimmedResult, null, 2),
    );
    return (
      `Saved the submitted PTX to \`${ptxFile}\` and the full result to ` +
      `\`${resultFile}\` in your workspace. Read either back with the read tool if needed.`
    );
  } catch (err) {
    return `(Could not save kernel artifacts: ${err.message})`;
  }
}

/** Matches the per-session kernel artifacts (kernel-<id>.ptx, kernel-result-<id>.json). */
const KERNEL_ARTIFACT_RE = /^kernel-\d+\.ptx$|^kernel-result-\d+\.json$/;

/**
 * Delete leftover kernel-<id>.ptx / kernel-result-<id>.json from a previous run
 * so each session starts clean (the submit counter restarts at 0). Returns the
 * number removed. Best-effort: never throws.
 */
export function cleanKernelArtifacts() {
  let removed = 0;
  let entries;
  try {
    entries = fs.readdirSync(WORKSPACE_DIR);
  } catch {
    return removed;
  }
  for (const name of entries) {
    if (!KERNEL_ARTIFACT_RE.test(name)) continue;
    try {
      fs.rmSync(path.join(WORKSPACE_DIR, name));
      removed++;
    } catch {
      // best-effort: ignore files we cannot remove
    }
  }
  return removed;
}

/** Build the single submit_ptx tool bound to a kernel id and session id. */
export function makeSubmitPtxTool(kernelId, tracker, sessionId) {
  // Per-session, monotonically increasing id for the saved kernel artifacts.
  let submitCount = 0;
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

      // Persist the candidate as a new best only when it beats every kernel
      // already saved for this kernel id (accumulate a history of records).
      if (result.passed) {
        const saved = saveBestKernel(
          kernelId,
          params.ptx,
          result.speedup_vs_triton,
          sessionId,
        );
        if (saved) {
          console.log(`${MAGENTA}[best]${RESET} ${GRAY}saved ${saved}${RESET}`);
        }
      }

      // Trim the echoed PTX out of the response, and persist both the PTX and
      // the full (trimmed) result to the workspace under a per-session id so the
      // agent can read them back if needed.
      const id = submitCount++;
      const trimmed = stripEchoedPtx(result);
      const saveNote = saveKernelArtifacts(
        `kernel-${id}.ptx`,
        `kernel-result-${id}.json`,
        params.ptx,
        trimmed,
      );

      const tag = result.passed ? `${GREEN}[ok]${RESET}` : `${RED}[fail]${RESET}`;
      const text = `${JSON.stringify(trimmed, null, 2)}\n\n${saveNote}`;
      console.log(`${tag} ${GRAY}${text}${RESET}`);

      return {
        content: [{ type: "text", text }],
        details: {},
      };
    },
  });
}
