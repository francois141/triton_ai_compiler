dim N;

array x[N];
array output[N];

output[i] = x[i] / (1 + exp(-x[i]));
