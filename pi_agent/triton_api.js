/**
 * Client for the Triton PTX environment HTTP server (see
 * `triton_ptx/triton_ptx/server.py`).
 *
 * The base URL defaults to the local server and can be overridden with the
 * `TRITON_PTX_URL` environment variable.
 */

const BASE_URL = (
  process.env.TRITON_PTX_URL || "http://127.0.0.1:8888"
).replace(/\/+$/, "");

/**
 * Error raised when the server returns a non-2xx response. Carries the HTTP
 * status and the server-provided error message.
 */
export class TritonApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "TritonApiError";
    this.status = status;
  }
}

/**
 * POST `body` as JSON to `path` and return the parsed JSON response.
 * Throws `TritonApiError` on a non-2xx status.
 */
async function post(path, body = {}) {
  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (cause) {
    throw new TritonApiError(0, `Cannot reach Triton server at ${BASE_URL}: ${cause.message}`);
  }

  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
      console.log(text)
    throw new TritonApiError(response.status, `Non-JSON response: ${text.slice(0, 600)}`);
  }

  if (!response.ok) {
    const message = (data && data.error) || `HTTP ${response.status}`;
    throw new TritonApiError(response.status, message);
  }
  return data;
}

/** Return the identifiers of every available kernel. */
export async function listKernels() {
  const data = await post("/list_kernels");
  return data.kernels;
}

/** Return whether a CUDA-capable GPU is available on the host. */
export async function isGpuAvailable() {
  const data = await post("/is_gpu_available");
  return data.gpu_available;
}

/** Return the Triton-generated reference PTX for a kernel. */
export async function dumpKernelPtx(kernelId) {
  const data = await post("/dump_kernel_ptx", { kernel_id: kernelId });
  return data.ptx;
}

/** Return the serializable data a client needs to prompt for a kernel. */
export async function getKernelData(kernelId) {
  return post("/get_kernel_data", { kernel_id: kernelId });
}

/**
 * Compile, verify, and benchmark a candidate; return the serializable result.
 * `payload` is a launch mapping such as
 * `{ ptx, num_threads_x, num_threads_y?, num_threads_z? }`.
 */
export async function evaluateCandidate(kernelId, payload) {
  return post("/evaluate_candidate", { kernel_id: kernelId, payload });
}

export { BASE_URL };
