dim N;

array gate[N];
array value[N];
array output[N];

output[i] = (gate[i] / (1 + exp(-gate[i]))) * value[i];
