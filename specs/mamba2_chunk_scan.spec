// Mamba-2 chunked scan pass: the carried state contribution C @ prev_states
// scaled by exp(dA), plus the intra-chunk contribution of the decayed,
// dt-weighted CB block against x. This variant of the kernel applies no
// causal mask - the decay exp(min(dA[row] - dA[k], 0)) is the only
// weighting - and its z gate and D skip path are both disabled, so neither
// appears here.
//
// The head and sequence axes are split the same way as in
// mamba2_chunk_state.spec, and for the same reason.

dim BATCH;
dim NC;
dim CHUNK;
dim NGROUPS;
dim RATIO;
dim HDIM;
dim DSTATE;

array cb[BATCH, NC, NGROUPS, CHUNK, CHUNK];
array x[BATCH, NC, CHUNK, NGROUPS, RATIO, HDIM];
array dt[BATCH, NGROUPS, RATIO, NC, CHUNK];
array dA_cumsum[BATCH, NGROUPS, RATIO, NC, CHUNK];
array C[BATCH, NC, CHUNK, NGROUPS, DSTATE];
array prev_states[BATCH, NC, NGROUPS, RATIO, HDIM, DSTATE];
array out[BATCH, NC, CHUNK, NGROUPS, RATIO, HDIM];

out[batch, chunk, row, group, head, channel] =
    exp(dA_cumsum[batch, group, head, chunk, row]) *
        sum(state in 0..DSTATE,
            C[batch, chunk, row, group, state] *
            prev_states[batch, chunk, group, head, channel, state]) +
    sum(k in 0..CHUNK,
        cb[batch, chunk, group, row, k] *
        dt[batch, group, head, chunk, k] *
        x[batch, chunk, k, group, head, channel] *
        exp(min(dA_cumsum[batch, group, head, chunk, row] -
            dA_cumsum[batch, group, head, chunk, k], 0)));
