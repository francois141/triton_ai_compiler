dim B;
dim D;

array x[B, D];
array weight[B, D];
array output[B, D];

output[row, column] = (x[row, column] * weight[row, column]) /
    sqrt(sum(k in 0..D, x[row, k] * x[row, k]) / D +
        0.0000009999999974752427);
