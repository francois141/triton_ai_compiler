// ForgettingAttention (ICLR 2025) forward pass, non-causal, with the
// softmax scale 1/sqrt(64) folded in. The forget gate adds the decay bias
// log_lambda[row] - log_lambda[key] to each score, so the weight of a key
// falls off with the cumulative log-decay between the two positions.
//
// The kernel works in base 2 with log2(e) folded into both the scale and
// the bias, which is the same softmax over the natural logits written
// here. Its max subtraction is left out: equal over the reals, and this
// form stays inside the sums-of-exponentials fragment the equivalence
// check is complete for. The logsumexp output l has no equation - it is
// m + log(sum), whose log volta carries as an uninterpreted atom.

dim Z;
dim H;
dim T;
dim D;

array q[Z, H, T, D];
array k[Z, H, T, D];
array v[Z, H, T, D];
array log_lambda[Z, H, T];
array o[Z, H, T, D];

o[batch, head, row, feature] =
    sum(key in 0..T,
        exp(0.125 * sum(e in 0..D,
            q[batch, head, row, e] * k[batch, head, key, e]) +
            log_lambda[batch, head, row] - log_lambda[batch, head, key]) *
        v[batch, head, key, feature]) /
    sum(key in 0..T,
        exp(0.125 * sum(e in 0..D,
            q[batch, head, row, e] * k[batch, head, key, e]) +
            log_lambda[batch, head, row] - log_lambda[batch, head, key]));
