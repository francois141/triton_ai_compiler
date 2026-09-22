dim M;
dim N;
dim K;

array a[M, K];
array b[K, N];
array d[M, N];
array c[M, N];

c[i, j] = (sum(k in 0..K, a[i, k] * b[k, j]) + d[i, j]) /
    (1 + exp(-(sum(k in 0..K, a[i, k] * b[k, j]) + d[i, j])));
