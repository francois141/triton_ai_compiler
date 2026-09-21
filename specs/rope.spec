dim H;
dim P;

array input[P, H];
array cos[P, H];
array sin[P, H];
array output[P, H];

output[half, feature] = input[half, feature] * cos[half, feature] +
    (2 * half - 1) * input[1 - half, feature] * sin[half, feature];
