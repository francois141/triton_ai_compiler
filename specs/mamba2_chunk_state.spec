// Mamba-2 chunked state pass. Each chunk's state is the dt-weighted,
// decayed outer product of that chunk's x and B blocks.
//
// The head axis is split into groups and heads within a group because B is
// shared by RATIO heads and the kernel selects it by integer division,
// which an index expression cannot express. The sequence axis is split into
// chunks and offsets within a chunk for the same reason. Each split is the
// outer half of a row-major pair, so the flat layout is unchanged.

dim BATCH;
dim NC;
dim CHUNK;
dim NGROUPS;
dim RATIO;
dim HDIM;
dim DSTATE;

array x[BATCH, NC, CHUNK, NGROUPS, RATIO, HDIM];
array b[BATCH, NC, CHUNK, NGROUPS, DSTATE];
array dt[BATCH, NGROUPS, RATIO, NC, CHUNK];
array dA_cumsum[BATCH, NGROUPS, RATIO, NC, CHUNK];
array states[BATCH, NC, NGROUPS, RATIO, HDIM, DSTATE];

states[batch, chunk, group, head, channel, state] =
    sum(k in 0..CHUNK,
        x[batch, chunk, k, group, head, channel] *
        b[batch, chunk, k, group, state] *
        dt[batch, group, head, chunk, k] *
        exp(min(dA_cumsum[batch, group, head, chunk, CHUNK - 1] -
            dA_cumsum[batch, group, head, chunk, k], 0)));
