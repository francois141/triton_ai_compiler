dim N;

array x[N];
array output[N];

output[i] = 0.5 * x[i] *
    (1 + tanh(0.7978845608028654 * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
