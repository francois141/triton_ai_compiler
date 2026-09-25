// FlashSinkhorn (ICML 2026) fused symmetric Sinkhorn step, with the
// operator's eps = 1, alpha = 0.5, damping = 1 and coord_scale = 2 folded
// in: f_out = (1 - alpha) * f_hat - alpha * eps * lse over the targets.
// The kernel's exp2/log2 pair cancels exactly over the reals, so it does
// not appear here. Its 1e-40 floor is left out too, but that one is not a
// no-op: the kernel floors the running-max-rescaled partial sum S, so the
// guard it applies to the true sum is 1e-40 * 2^m for a data-dependent m,
// which the spec has no way to name.
//
// Only the f update is stated. The launch splits its flat grid in two -
// the first half updates f, the second updates g - and the analysis
// executes CTA 0 alone, so g_out is never written under this
// configuration and an equation for it would have nothing to check
// against.

dim N;
dim M;
dim D;

array x[N, D];
array y[M, D];
array f_hat[N];
array g_hat[M];
array log_b[M];
array f_out[N];

f_out[i] = 0.5 * f_hat[i] - 0.5 * log(sum(j in 0..M,
    exp(2 * sum(e in 0..D, x[i, e] * y[j, e]) + g_hat[j] + log_b[j])));
