// FlashSinkhorn (ICML 2026) fused symmetric Sinkhorn step, with the
// operator's eps = 1, alpha = 0.5, damping = 1 and coord_scale = 2 folded
// in: f_out = (1 - alpha) * f_hat - alpha * eps * lse over the targets.
// The kernel's exp2/log2 pair and its 1e-40 floor under the log are exact
// no-ops over the reals, so neither appears here.
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
