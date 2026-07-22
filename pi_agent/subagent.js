/**
 * The `spawn_subagent` tool: lets the lead agent delegate a focused task to an
 * isolated nested agent session.
 *
 * The subagent runs in the SAME workspace/jail/skill as the parent with a scoped
 * toolset (read-only tools + submit_ptx, but NOT spawn_subagent, so it cannot
 * recurse). Its intermediate steps stay in its own context; only its final
 * message is returned to the parent, which keeps the parent's context clean.
 *
 * There are no specialized agent types: the parent supplies complete, self-
 * contained instructions per call.
 */

import {
  createAgentSession,
  DefaultResourceLoader,
  getAgentDir,
  SessionManager,
  defineTool,
} from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

import {
  makeWorkspaceJail,
  READ_ONLY_TOOL_NAMES,
  SKILL_DIRS,
  TRITON_SOURCE_FILE,
  WORKSPACE_DIR,
} from "./tool.js";
import { promptUntilSettled } from "./agent.js";
import { MAGENTA, RED, RESET } from "./tui.js";

const SUBAGENT_SYSTEM_PROMPT =
  "You are a subagent spawned by a lead GPU-optimization agent to complete one " +
  "focused task -- typically searching the PTX ISA documentation or fixing a " +
  "PTX compilation error. You work in the same workspace as the lead agent: " +
  "read files with read/ls/grep/find (confined to the workspace, which includes " +
  "the PTX ISA reference skill), and compile/verify/benchmark PTX candidates " +
  "with submit_ptx. The kernel's original Triton source is saved at " +
  `\`${TRITON_SOURCE_FILE}\` in the workspace. Follow the instructions you are ` +
  "given, use your tools to do the work, and finish with a single concise final " +
  "message containing exactly the result the lead agent needs -- no filler. Your " +
  "intermediate steps are hidden from the lead agent; only your final message is " +
  "returned.";

// Cap the returned answer so a runaway subagent cannot flood the parent context.
const MAX_ANSWER_BYTES = 200 * 1024;

/**
 * Build the spawn_subagent tool.
 *
 * @param {object} deps
 * @param {import("@earendil-works/pi-coding-agent").ModelRuntime} deps.modelRuntime
 * @param {string} deps.sessionsDir            Directory for child session JSONL files.
 * @param {string} deps.parentSessionId        Linked as each child's parentSession.
 * @param {object} deps.submitPtx              The parent's submit_ptx tool instance,
 *   reused so the kernel-<id> counter stays globally monotonic across parent and child.
 */
export function makeSpawnSubagentTool({
  modelRuntime,
  sessionsDir,
  parentSessionId,
  submitPtx,
}) {
  // The scaffold prompt is fixed, so one loader serves every spawn (only the
  // per-call task message differs). Built lazily and cached.
  let loaderPromise = null;
  function getLoader() {
    if (!loaderPromise) {
      const loader = new DefaultResourceLoader({
        cwd: WORKSPACE_DIR,
        agentDir: getAgentDir(),
        systemPromptOverride: () => SUBAGENT_SYSTEM_PROMPT,
        appendSystemPromptOverride: () => [],
        additionalSkillPaths: SKILL_DIRS,
        extensionFactories: [makeWorkspaceJail()],
      });
      loaderPromise = loader.reload().then(() => loader);
    }
    return loaderPromise;
  }

  let spawnCount = 0;

  return defineTool({
    name: "spawn_subagent",
    label: "Spawn subagent",
    description:
      "Delegate a focused task to an isolated subagent with its own fresh " +
      "context. Provide complete, self-contained instructions: the subagent " +
      "does not see this conversation. It can read the workspace and PTX ISA " +
      "docs (read/ls/grep/find) and compile/verify PTX (submit_ptx), and it " +
      "returns only its final message. Use it to keep large intermediate work " +
      "-- documentation searches, compile-error debugging -- out of your own " +
      "context.",
    parameters: Type.Object({
      instructions: Type.String({
        description:
          "Complete, self-contained instructions for the subagent: what to do " +
          "and exactly what to return. The subagent has no other context.",
      }),
    }),
    execute: async (_toolCallId, params) => {
      const id = spawnCount++;

      // Persist the child session, linked to the parent for later analysis.
      const childManager = SessionManager.create(WORKSPACE_DIR, sessionsDir, {
        parentSession: parentSessionId,
      });
      console.log(
        `\n${MAGENTA}[subagent ${id}] spawned -> ${childManager.getSessionFile()}${RESET}`,
      );

      let child;
      try {
        const loader = await getLoader();
        ({ session: child } = await createAgentSession({
          cwd: WORKSPACE_DIR,
          customTools: [submitPtx],
          // Same tools as the parent MINUS spawn_subagent -> no recursion.
          tools: ["submit_ptx", ...READ_ONLY_TOOL_NAMES],
          thinkingLevel: "medium",
          modelRuntime,
          resourceLoader: loader,
          sessionManager: childManager,
        }));
      } catch (err) {
        const message = `Subagent failed to start: ${err.message}`;
        console.log(`${RED}[subagent ${id}] ${message}${RESET}`);
        return { content: [{ type: "text", text: message }], details: { error: message } };
      }

      try {
        // Stream the subagent's reasoning and tool calls the same way the lead
        // agent does (labelled so nested output is distinguishable), and resume
        // after any truncated turn, just like the lead agent.
        await promptUntilSettled(child, params.instructions, {
          label: `[subagent ${id}]`,
        });
        let answer = child.getLastAssistantText() ?? "";
        if (!answer.trim()) {
          answer = "(The subagent finished without producing a final message.)";
        } else if (Buffer.byteLength(answer, "utf8") > MAX_ANSWER_BYTES) {
          answer = `${answer.slice(0, MAX_ANSWER_BYTES)}\n\n[subagent output truncated]`;
        }
        console.log(`${MAGENTA}[subagent ${id}] done${RESET}`);
        return { content: [{ type: "text", text: answer }], details: {} };
      } catch (err) {
        const message = `Subagent run failed: ${err.message}`;
        console.log(`${RED}[subagent ${id}] ${message}${RESET}`);
        return { content: [{ type: "text", text: message }], details: { error: message } };
      } finally {
        child?.dispose();
      }
    },
  });
}
