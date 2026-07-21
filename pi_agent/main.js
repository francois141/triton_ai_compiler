/**
 * CLI entry point for the Triton PTX optimization agent.
 *
 * Usage:
 *     node main.js [KernelId]
 *
 * With a KernelId, optimize that single kernel. With no argument, list every
 * available kernel and optimize each one in turn (a fresh agent per kernel,
 * run sequentially).
 *
 * Environment variables:
 *     TRITON_PTX_URL   Base URL of the Triton PTX server (see triton_api.js).
 */

import { runAgentOnKernel } from "./agent.js";
import { listKernels } from "./triton_api.js";
import { GRAY, MAGENTA, RED, RESET } from "./tui.js";

async function main() {
  const kernelId = process.argv[2];

  if (kernelId) {
    await runAgentOnKernel(kernelId);
    return;
  }

  // No kernel given: optimize all of them sequentially.
  const kernels = await listKernels();
  console.log(
    `${MAGENTA}Optimizing all ${kernels.length} kernels sequentially.${RESET}`,
  );

  const results = [];
  for (const [index, kernel] of kernels.entries()) {
    console.log(
      `\n${MAGENTA}=== [${index + 1}/${kernels.length}] ${kernel} ===${RESET}`,
    );
    try {
      const bestSpeedup = await runAgentOnKernel(kernel);
      results.push({ kernel, bestSpeedup });
    } catch (err) {
      // One kernel failing should not abort the whole batch.
      console.error(
        `${RED}[error]${RESET} ${GRAY}Kernel ${kernel} failed: ` +
          `${err.stack || err.message}${RESET}`,
      );
      results.push({ kernel, bestSpeedup: 0, error: err.message });
    }
  }

  // Print a final summary of every kernel's best passing speedup.
  console.log(`\n${MAGENTA}=== Summary ===${RESET}`);
  for (const { kernel, bestSpeedup, error } of results) {
    const status = error
      ? `${RED}error: ${error}${RESET}`
      : bestSpeedup > 0
        ? `${bestSpeedup.toFixed(3)}x`
        : "none";
    console.log(`${GRAY}${kernel}:${RESET} ${status}`);
  }
}

main().catch((err) => {
  console.error(`${RED}Fatal:${RESET} ${err.stack || err.message}`);
  process.exitCode = 1;
});
