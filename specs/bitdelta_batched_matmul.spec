// Batched BitDelta (NeurIPS 2024) binary-weight GEMM: every batch entry
// computes c = a * sign(b), where each weight contributes +1 or -1. The
// kernel reads those signs from int32 words holding 32 bits each, and
// decodes bit t of the word as 2 * bit - 1; the spec language has no
// bitwise operators, so the decode is written over the unpacked weight
// matrix as sign(b[batch, k, j]).

dim BATCH;
dim M;
dim N;
dim K;

array a[BATCH, M, K];
array b[BATCH, K, N];
array c[BATCH, M, N];

c[batch, i, j] = sum(k in 0..K, a[batch, i, k] * sign(b[batch, k, j]));
