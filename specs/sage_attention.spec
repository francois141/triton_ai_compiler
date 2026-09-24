// SageAttention (ICLR 2025) forward pass over int8 queries and keys, one
// scale per query block and per key block. Grouped-query attention: each
// of the KVH key heads is shared by G query heads, and the block axes are
// split out (MB by BM queries, NB by BN keys) because the kernel derives
// the scale and key-head indices by integer division, which an index
// expression cannot express. Every split axis is the outer half of a
// row-major pair, so the flat layout matches the kernel's.
//
// The kernel exponentiates base 2, which is a softmax over ln(2) * logits.

dim Z;
dim KVH;
dim G;
dim MB;
dim BM;
dim NB;
dim BN;
dim D;

array q[Z, KVH, G, MB, BM, D];
array k[Z, KVH, NB, BN, D];
array v[Z, KVH, NB, BN, D];
array q_scale[Z, KVH, G, MB];
array k_scale[Z, KVH, NB];
array o[Z, KVH, G, MB, BM, D];

o[batch, kv_head, group, row_block, row, feature] =
    sum(key_block in 0..NB, sum(key in 0..BN,
        exp(0.6931471805599453 *
            sum(e in 0..D, q[batch, kv_head, group, row_block, row, e] *
                k[batch, kv_head, key_block, key, e]) *
            q_scale[batch, kv_head, group, row_block] *
            k_scale[batch, kv_head, key_block]) *
        v[batch, kv_head, key_block, key, feature])) /
    sum(key_block in 0..NB, sum(key in 0..BN,
        exp(0.6931471805599453 *
            sum(e in 0..D, q[batch, kv_head, group, row_block, row, e] *
                k[batch, kv_head, key_block, key, e]) *
            q_scale[batch, kv_head, group, row_block] *
            k_scale[batch, kv_head, key_block])));
