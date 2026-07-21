/**
 * Minimal Pi-SDK agent that optimizes a single Triton kernel at the PTX level.
 *
 * This module exposes `runAgentOnKernel(kernelId)`; the CLI entry point that
 * drives it (single kernel or all kernels in a loop) lives in main.js.
 *
 * Environment variables:
 *     TRITON_PTX_URL   Base URL of the Triton PTX server (see triton_api.js).
 */

import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  ModelRuntime,
  SessionManager,
} from "@earendil-works/pi-coding-agent";

import { getKernelData } from "./triton_api.js";
import {
  cleanKernelArtifacts,
  makeSubmitPtxTool,
  makeWorkspaceJail,
  READ_ONLY_TOOL_NAMES,
  saveTritonSource,
  SKILL_DIRS,
  TRITON_SOURCE_FILE,
  WORKSPACE_DIR,
} from "./tool.js";
import { makeSpawnSubagentTool } from "./subagent.js";
import { GRAY, MAGENTA, RED, RESET, renderSessionOutput } from "./tui.js";

const SYSTEM_PROMPT =
  "You are an expert GPU engineer optimizing a single Triton kernel at the " +
  "NVIDIA PTX (assembly) level. You are given the Triton source, the PTX " +
  "signature, the launch configuration, and a PTX entry template to fill " +
  "in.\n\n" +
  "Your goal is to produce PTX that is functionally identical to the Triton " +
  "kernel but runs faster. Call submit_ptx to compile, verify, and " +
  "benchmark a candidate; it returns whether it compiled, whether it is " +
  "correct, its p50 latency, and its speedup versus Triton. Iterate: read " +
  "the result, fix errors, and keep trying to improve the speedup while " +
  "staying correct. Only correct (passing) candidates count. When you are " +
  "satisfied that you cannot improve further, stop and summarize your best " +
  "result." +
  "You can use your tools to read the PTX documentation, provided as a skill." +
  "Your tools can only access data in the `agent_workspace`.\n\n" +
  "You can also call spawn_subagent to delegate a focused task to an isolated " +
  "subagent that has the same tools but a fresh, separate context. Use it to " +
  "keep intermediate work out of your own context -- for example, " +
  "searching the PTX documentation for a specific answer, debugging a " +
  "compilation error, or testing an hypothesis -- since only the subagent's final message is returned to " +
  "you. Give it complete, self-contained instructions, including exactly what " +
  "you want it to report back.";

// --- Environment context ---

/** Match a parameter annotation string against Triton's constexpr type. */
function isConstexprAnnotation(annotation) {
  const text = String(annotation).toLowerCase();
  return (
    text === "constexpr" ||
    text.endsWith(".constexpr") ||
    text.includes("triton.language.core.constexpr") ||
    (text.includes("triton.language") && text.includes("constexpr"))
  );
}

/**
 * Build the "Operator Constexpr Values" section. These tl.constexpr parameters
 * are folded into the PTX at compile time by Triton, so they are absent from
 * the PTX signature; the model must hardcode these exact values.
 */
function constexprValuesBlock(data) {
  const constexprParams = (data.parameters ?? []).filter((param) =>
    isConstexprAnnotation(param.annotation),
  );
  if (constexprParams.length === 0) {
    return "Operator constexpr values:\nNone.";
  }

  const constexprValues = data.constexpr_values ?? {};
  const lines = constexprParams.map((param) =>
    param.name in constexprValues
      ? `- ${param.name}: ${JSON.stringify(constexprValues[param.name])}`
      : `- ${param.name}: unavailable from operator constexpr_values`,
  );

  // It seems we have a bug: the evaluator uses different values.
  // return (
  //   "Operator constexpr values (fixed at compile time; folded into the " +
  //   "kernel and absent from the signature). Use these exact values when " +
  //   "writing indexing logic:\n" +
  //   lines.join("\n")
  // );

  return (
    "Operator constexpr values (fixed at compile time; folded into the " +
    "kernel and absent from the signature). Those values may change, only assume they are multiple of 32 \n" +
    lines.join("\n")
  );
}

/**
 * Build the PTX entry template. Gives the model the exact entry shape --
 * directives, kernel name, and the real compiled parameter list -- with an
 * empty body to fill, standing in for the reference PTX.
 */
function ptxEntryTemplate(data) {
  const system = data.system;
  const params = data.ptx_signature
    .map((param) => `    .param ${param.ptx_type} ${param.name}`)
    .join(",\n");

  return (
    "PTX entry template. Use this exact entry shape and fill the body with " +
    "your PTX (any argument name containing `_ptr` is a pointer to float32 " +
    "data):\n" +
    "```ptx\n" +
    `.version ${system.version}\n` +
    `.target ${system.target}\n` +
    `.address_size ${system.address_size}\n\n` +
    `.visible .entry ${data.kernel_name}(\n` +
    `${params}\n` +
    ")\n" +
    "{\n" +
    "   // TODO: Fill this part with your own PTX\n" +
    "}\n" +
    "```"
  );
}

/** Assemble the kernel context handed to the model in the first turn. */
async function buildContextMessage(kernelId) {
  const data = await getKernelData(kernelId);

  // Save the Triton source in the workspace so subagents (which don't share this
  // context) can read it, and point the model at it.
  saveTritonSource(data.source);

  const system = data.system;
  const signature = data.ptx_signature
    .map((param) => `  ${param.name}: ${param.ptx_type}`)
    .join("\n");
  const defaultThreadsX = data.num_warps * 32;

  return (
    `Kernel to optimize: ${kernelId}\n` +
    `PTX kernel name: ${data.kernel_name}\n\n` +
    `Target: ${system.target} | PTX version: ${system.version} | ` +
    `address size: ${system.address_size}\n` +
    `num_warps: ${data.num_warps} ` +
    `(reference launch uses num_threads_x = ${defaultThreadsX})\n\n` +
    "PTX parameter signature:\n" +
    `${signature}\n\n` +
    `${constexprValuesBlock(data)}\n\n` +
    `Triton source (also saved to \`${TRITON_SOURCE_FILE}\` in your workspace):\n` +
    "```python\n" +
    `${data.source}\n` +
    "```\n\n" +
    `${ptxEntryTemplate(data)}\n\n` +
    "Submit an optimized PTX candidate with submit_ptx."
  );
}

// --- Agent ---

/**
 * Run the optimization agent on a single kernel end-to-end.
 *
 * @param {string} kernelId  Identifier of the kernel to optimize.
 * @returns {Promise<number>} The best passing speedup vs Triton (0 if none).
 */
export async function runAgentOnKernel(kernelId) {
  console.log(`Optimizing kernel ${JSON.stringify(kernelId)} with the Pi SDK`);

  // Clear kernel-<id>.ptx / kernel-result-<id>.json from any previous run so
  // this session's submit counter starts from a clean workspace.
  const cleaned = cleanKernelArtifacts();
  if (cleaned > 0) {
    console.log(`${GRAY}Cleared ${cleaned} kernel artifact(s) from a previous run.${RESET}`);
  }

  const contextMessage = await buildContextMessage(kernelId);
  const tracker = { bestSpeedup: 0 };

  // Persist the run as an append-only JSONL session tree so it can be analyzed
  // later. Stored under pi_agent/sessions/ (gitignore it to keep runs local).
  // Inspect with `pi --session <file>` or by reading the JSONL directly.
  const sessionsDir = fileURLToPath(new URL("./sessions", import.meta.url));
  const sessionManager = SessionManager.create(WORKSPACE_DIR, sessionsDir);
  console.log(`${MAGENTA}Session log: ${sessionManager.getSessionFile()}${RESET}`);

  // The session id is embedded in saved best-kernel filenames so each record
  // links back to this run's session.
  const submitPtx = makeSubmitPtxTool(
    kernelId,
    tracker,
    sessionManager.getSessionId(),
  );

  const loader = new DefaultResourceLoader({
    // Run tools out of the workspace so relative paths resolve there and the
    // read-only tools default to it; the jail below blocks anything outside it.
    cwd: WORKSPACE_DIR,
    agentDir: getAgentDir(),
    systemPromptOverride: () => SYSTEM_PROMPT,
    // Avoid appending any APPEND_SYSTEM.md discovered from disk.
    appendSystemPromptOverride: () => [],
    // Advertise the bundled skills (name + description) to the model. Pi appends
    // the skills section to our custom system prompt because the read tool is
    // enabled; the model then reads SKILL.md on demand via the read-only tools.
    additionalSkillPaths: SKILL_DIRS,
    // Confine the built-in read-only tools to WORKSPACE_DIR (skills live there).
    extensionFactories: [makeWorkspaceJail()],
  });
  await loader.reload();

  // Load the model list from the local models.json (next to this file) so its
  // per-model settings -- notably a raised maxTokens -- take effect. Credentials
  // still come from the default agent dir / the $EPFL_API_KEY env var referenced
  // in models.json. Model selection is left to auto-discovery for now.
  const modelsPath = fileURLToPath(new URL("./models.json", import.meta.url));
  const modelRuntime = await ModelRuntime.create({
    modelsPath,
    authPath: path.join(getAgentDir(), "auth.json"),
  });

  // Lets the lead agent delegate focused tasks to isolated subagents. Reuses the
  // parent's submitPtx (shared kernel counter) and modelRuntime; child sessions
  // are saved under sessionsDir linked to this run's session.
  const spawnSubagent = makeSpawnSubagentTool({
    modelRuntime,
    sessionsDir,
    parentSessionId: sessionManager.getSessionId(),
    submitPtx,
  });

  const { session } = await createAgentSession({
    cwd: WORKSPACE_DIR,
    customTools: [submitPtx, spawnSubagent],
    // Enable submit_ptx and spawn_subagent plus the built-in read-only tools
    // (read/ls/grep/find), which the workspace jail confines to WORKSPACE_DIR.
    // No write/edit/bash.
    tools: ["submit_ptx", "spawn_subagent", ...READ_ONLY_TOOL_NAMES],
    // Stream reasoning too, the way the Pi TUI does (ignored by models that do
    // not support thinking). Adjust with model selection later.
    thinkingLevel: "medium",
    modelRuntime,
    resourceLoader: loader,
    sessionManager,
  });

  // TUI-style streaming: render reasoning (grey), answer text (cyan), tool
  // calls, and compaction notices live. Shared with subagents via render.js.
  const { getLastStopReason } = renderSessionOutput(session);

  try {
    // prompt() resolves only after the full run (all turns + tool calls) ends.
    await session.prompt(contextMessage);
  } finally {
    session.dispose();
  }

  const best =
    tracker.bestSpeedup > 0 ? tracker.bestSpeedup.toFixed(3) + "x" : "none";
  console.log(`\n${GRAY}Done. Best passing speedup vs Triton: ${best}.${RESET}`);
  console.log(`${MAGENTA}Session saved to: ${sessionManager.getSessionFile()}${RESET}`);
  if (getLastStopReason() === "length") {
    console.log(
      `${RED}Note:${RESET} ${GRAY}the run ended on a truncated turn (output-token ` +
        `limit), not because the model chose to stop.${RESET}`,
    );
  }

  return tracker.bestSpeedup;
}
