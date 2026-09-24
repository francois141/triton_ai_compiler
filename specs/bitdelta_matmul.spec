// BitDelta (NeurIPS 2024) binary-weight GEMM: c = a * sign(b), where every
// weight contributes +1 or -1. The kernel reads those signs from int32
// words holding 32 bits each, and decodes bit t of the word as
// 2 * bit - 1; the spec language has no bitwise operators, so the decode
// is written over the unpacked weight matrix as sign(b[k, j]).

dim M;
dim N;
dim K;

array a[M, K];
array b[K, N];
array c[M, N];

c[i, j] = sum(k in 0..K, a[i, k] * sign(b[k, j]));
