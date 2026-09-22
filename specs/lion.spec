dim N;

array grad[N];
array p[N];
array exp_avg[N];

p[i] = p[i] * 0.9999989867210388 -
    0.00009999999747378752 *
        sign((exp_avg[i] - grad[i]) * 0.8999999761581421 + grad[i]);
exp_avg[i] = (exp_avg[i] - grad[i]) * 0.9900000095367432 + grad[i];
