#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include "assert.h"

#define CUDA_CHECK(call)                                                               \
    do                                                                                 \
    {                                                                                  \
        const cudaError_t error = (call);                                              \
        if (error != cudaSuccess)                                                      \
        {                                                                              \
            std::cerr << cudaGetErrorString(error) << '\n';                            \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                             \
    do                                                                                 \
    {                                                                                  \
        const cublasStatus_t status = (call);                                          \
        if (status != CUBLAS_STATUS_SUCCESS)                                           \
        {                                                                              \
            std::cerr << "cuBLAS error " << static_cast<int>(status) << '\n';          \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

constexpr int MATRIX_SIZE = 4096;
constexpr int WARMUP_ITERATIONS = 50;
constexpr int BENCHMARK_ITERATIONS = 500;

constexpr int MMA_TILE_M = 128;
constexpr int MMA_TILE_N = 256;
constexpr int MMA_TILE_K = 32;
constexpr int WARP_COUNT_M = 2;
constexpr int WARP_COUNT_N = 4;
constexpr int WARP_MMA_ITERATIONS_M = 4;
constexpr int WARP_MMA_ITERATIONS_N = 8;
constexpr int MMA_STAGES = 3;
constexpr int WARP_GEMM_ITERATIONS = 2;
constexpr int ASYNC_COPY_ITERATIONS_A = 2;
constexpr int ASYNC_COPY_ITERATIONS_B = 4;
constexpr int ACCESSES_PER_GROUP_A = 1;
constexpr int ACCESSES_PER_GROUP_B = 2;
constexpr int EPILOGUE_WARP_COUNT_M = 2;
constexpr int EPILOGUE_WARP_COUNT_N = 4;
constexpr int EPILOGUE_ITERATIONS = 8;
constexpr int THREAD_COUNT = 256;
constexpr int SHARED_A_STRIDE = MMA_STAGES * MMA_TILE_K;
constexpr int SHARED_B_STRIDE = MMA_TILE_N;
constexpr int SHARED_A_ELEMENTS = MMA_TILE_M * SHARED_A_STRIDE;
constexpr int SHARED_B_ELEMENTS = MMA_STAGES * MMA_TILE_K * SHARED_B_STRIDE;
constexpr int SHARED_MEMORY_BYTES =
    (SHARED_A_ELEMENTS + SHARED_B_ELEMENTS) * sizeof(__half);

struct KernelParams
{
    const __half *matrix_a;
    const __half *matrix_b;
    __half *matrix_d;
    int rows;
    int columns;
    int inner;
    int stride_a;
    int stride_b;
    int stride_d;
};

__device__ __forceinline__ void commit_async_copy_group()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int pending_groups>
__device__ __forceinline__ void wait_for_async_copy_groups()
{
    asm volatile("cp.async.wait_group %0;\n" : : "n"(pending_groups));
}

__device__ __forceinline__ void wait_for_all_async_copies()
{
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void async_copy_zfill_16(void *destination,
                                                    const void *source, bool is_valid)
{
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(destination));
    const int source_bytes = is_valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n"
                 :
                 : "r"(shared_address), "l"(source), "r"(source_bytes));
}

__device__ __forceinline__ void mma_sync_m16n8k16(float (&d)[4], const __half (&a)[8],
                                                  const __half (&b)[4],
                                                  const float (&c)[4])
{
    const uint32_t *a_registers = reinterpret_cast<const uint32_t *>(&a);
    const uint32_t *b_registers = reinterpret_cast<const uint32_t *>(&b);
    const float *c_registers = reinterpret_cast<const float *>(&c);
    float *d_registers = reinterpret_cast<float *>(&d);

    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=f"(d_registers[0]), "=f"(d_registers[1]), "=f"(d_registers[2]),
                   "=f"(d_registers[3])
                 : "r"(a_registers[0]), "r"(a_registers[1]), "r"(a_registers[2]),
                   "r"(a_registers[3]), "r"(b_registers[0]), "r"(b_registers[1]),
                   "f"(c_registers[0]), "f"(c_registers[1]), "f"(c_registers[2]),
                   "f"(c_registers[3]));
}

template <bool TRANSPOSE>
__device__ __forceinline__ void load_shared_matrix_x4(uint32_t *destination,
                                                      const void *source)
{
    const unsigned shared_address =
        static_cast<unsigned>(__cvta_generic_to_shared(source));
    if constexpr (TRANSPOSE)
    {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                     "{%0,%1,%2,%3}, [%4];\n"
                     : "=r"(destination[0]), "=r"(destination[1]), "=r"(destination[2]),
                       "=r"(destination[3])
                     : "r"(shared_address));
    }
    else
    {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                     "{%0,%1,%2,%3}, [%4];\n"
                     : "=r"(destination[0]), "=r"(destination[1]), "=r"(destination[2]),
                       "=r"(destination[3])
                     : "r"(shared_address));
    }
}

struct WarpTileLoaderA
{
    const uint4 *pointer;
    int stride;
    int sections;
    int byte_offset;
    int k_group;

    __device__ WarpTileLoaderA(const __half *base, int reference_stride, int lane)
        : pointer(reinterpret_cast<const uint4 *>(base)), stride(reference_stride / 4),
          sections(reference_stride / 32), byte_offset(0), k_group(0)
    {
        const int partition = lane & 1;
        const int access_strided = (lane & 15) / 2;
        const int access_contiguous = partition * 4 + ((lane >> 4) ^ ((lane & 7) / 2));
        byte_offset = (access_contiguous + access_strided * stride) * 16;
    }

    __device__ void add_tile_offset(int strided_tile, int contiguous_tile)
    {
        const int whole_tiles = contiguous_tile / 2;
        const int k_group_delta = contiguous_tile % 2;
        byte_offset ^= k_group_delta * 32;
        pointer += strided_tile * stride * 32 + whole_tiles * stride / sections;
    }

    __device__ void set_kgroup_index(int group) { k_group = group & 1; }

    __device__ void load(__half (&fragment)[32]) const
    {
        auto *registers = reinterpret_cast<uint32_t *>(fragment);
#pragma unroll
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            const char *source =
                reinterpret_cast<const char *>(pointer + iteration * 8 * stride);
            load_shared_matrix_x4<false>(registers + iteration * 4,
                                         source + byte_offset);
        }
    }

    __device__ void advance()
    {
        byte_offset ^= 32;
        if (++k_group == 2)
        {
            k_group = 0;
            add_tile_offset(0, 2);
        }
    }
};

struct WarpTileLoaderB
{
    const uint4 *pointers[4];
    int stride;
    int byte_offset;

    __device__ WarpTileLoaderB(const __half *base, int reference_stride, int lane)
        : stride(reference_stride / 8), byte_offset(0)
    {
        const int quad_quad = lane >> 4;
        const int lane_in_quad = lane & 3;
        const int lane_in_quad_pair = lane & 7;
        const int lane_in_quad_quad = lane & 15;
        const auto *base_pointer = reinterpret_cast<const uint4 *>(base);
#pragma unroll
        for (int index = 0; index < 4; ++index)
        {
            const int partition = (lane_in_quad_pair >> 2) ^ (index >> 1);
            const int access_contiguous =
                partition * 4 + ((quad_quad + ((index & 1) << 1)) ^ lane_in_quad);
            pointers[index] =
                base_pointer + access_contiguous + lane_in_quad_quad * stride;
        }
    }

    __device__ void add_tile_offset(int strided_tile, int contiguous_tile)
    {
        byte_offset += (strided_tile * 16 * stride * 8 + contiguous_tile * 64) * 2;
    }

    __device__ void load(__half (&fragment)[32]) const
    {
        auto *registers = reinterpret_cast<uint32_t *>(fragment);
#pragma unroll
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            const char *source = reinterpret_cast<const char *>(pointers[iteration]);
            load_shared_matrix_x4<true>(registers + iteration * 4,
                                        source + byte_offset);
        }
    }

    __device__ void advance() { byte_offset += 256 * stride; }
};

__device__ __forceinline__ int shared_layout_offset(int contiguous, int strided,
                                                    int stride, int crosswise)
{
    constexpr int elements_per_access = 8;
    constexpr int tile_contiguous = 8;
    constexpr int partition_contiguous = 4;
    constexpr int partition_strided = 4;
    const int factor = 64 / crosswise;
    const int tile_strided = max(tile_contiguous / factor, 4);
    const int vector_contiguous = contiguous / elements_per_access;
    const int vector_strided = strided / factor;
    const int tile_contiguous_index = vector_contiguous / (tile_contiguous / factor);
    const int tile_contiguous_residual =
        vector_contiguous % (tile_contiguous / factor) +
        (strided % factor) * (tile_contiguous / factor);
    const int tile_strided_residual = vector_strided % tile_strided;
    const int partition_contiguous_index =
        tile_contiguous_residual / partition_contiguous;
    const int partition_strided_index = tile_strided_residual / partition_strided;
    const int contiguous_residual = tile_contiguous_residual % partition_contiguous;
    const int strided_residual = tile_strided_residual % partition_strided;
    const int permuted_contiguous = contiguous_residual ^ (strided_residual % 4);
    const int permuted_partition =
        partition_contiguous_index ^ (partition_strided_index % 2);
    const int element_contiguous =
        (tile_contiguous_index * tile_contiguous +
         permuted_partition * partition_contiguous + permuted_contiguous) *
            elements_per_access +
        contiguous % elements_per_access;
    return element_contiguous + vector_strided * stride * factor;
}

struct SharedMemoryStoreA
{
    uint4 *pointers[2];
    int byte_offset;
    int iteration;
    int stage_increment;

    __device__ SharedMemoryStoreA(__half *base, int stride, int thread_id)
        : byte_offset(0), iteration(0)
    {
        const int warp = thread_id / 32;
        const int lane = thread_id % 32;
        const int contiguous = (lane % 4) * 8;
        const int strided = warp * 16 + lane / 4;
#pragma unroll
        for (int index = 0; index < 2; ++index)
        {
            const int offset =
                shared_layout_offset(contiguous, strided + index * 8, stride, 32);
            pointers[index] = reinterpret_cast<uint4 *>(base + offset);
        }
        const int iterator_stride = stride / 4;
        const int sections = stride / 32;
        stage_increment = iterator_stride * 8 / sections * sizeof(__half);
    }

    __device__ void set_iteration_index(int index) { iteration = index; }
    __device__ void *get() const
    {
        return reinterpret_cast<char *>(pointers[iteration & 1]) + byte_offset;
    }
    __device__ void advance() { iteration = (iteration + 1) & 1; }
    __device__ void add_tile_offset(int, int contiguous_tile)
    {
        byte_offset += contiguous_tile * stage_increment;
    }
};

struct SharedMemoryStoreB
{
    uint4 *pointer;
    int byte_offset;
    int iteration;
    int stage_increment;

    __device__ SharedMemoryStoreB(__half *base, int stride, int thread_id)
        : byte_offset(0), iteration(0)
    {
        const int warp = thread_id / 32;
        const int lane = thread_id % 32;
        const int contiguous = (lane % 8) * 8;
        const int strided = warp * 4 + lane / 8;
        const int offset = shared_layout_offset(contiguous, strided, stride, 64);
        pointer = reinterpret_cast<uint4 *>(base + offset);
        stage_increment = 32 * stride * sizeof(__half);
    }

    __device__ void set_iteration_index(int index) { iteration = index; }
    __device__ void *get() const
    {
        return reinterpret_cast<char *>(pointer + iteration * 8) + byte_offset;
    }
    __device__ void advance() { iteration = (iteration + 1) & 3; }
    __device__ void add_tile_offset(int strided_tile, int)
    {
        byte_offset += strided_tile * stage_increment;
    }
};

struct GlobalTileLoaderA
{
    const __half *pointer;
    int leading_dimension;
    int extent_columns;
    int column;
    unsigned valid_rows;

    __device__ GlobalTileLoaderA(const __half *source, int stride, int rows,
                                 int columns, int thread_id, int row, int column)
        : leading_dimension(stride), extent_columns(columns), valid_rows(0)
    {
        const int warp = thread_id / 32;
        const int lane = thread_id % 32;
        const int thread_row = row + warp * 16 + lane / 4;
        this->column = column + (lane % 4) * 8;
        pointer = source + thread_row * stride + this->column;
        valid_rows = static_cast<unsigned>(thread_row < rows) |
                     (static_cast<unsigned>(thread_row + 8 < rows) << 1);
    }

    __device__ const void *get(int index) const
    {
        return pointer + index * 8 * leading_dimension;
    }
    __device__ bool valid(int index) const
    {
        return ((valid_rows >> index) & 1U) && column < extent_columns;
    }
    __device__ void add_tile_offset(int row_tiles, int column_tiles)
    {
        const int row_offset = row_tiles * 128;
        const int column_offset = column_tiles * 32;
        pointer += row_offset * leading_dimension + column_offset;
        column += column_offset;
    }
};

struct GlobalTileLoaderB
{
    const __half *pointer;
    int leading_dimension;
    int extent_rows;
    int row;
    unsigned valid_columns;

    __device__ GlobalTileLoaderB(const __half *source, int stride, int rows,
                                 int columns, int thread_id, int row, int column)
        : leading_dimension(stride), extent_rows(rows), valid_columns(0)
    {
        const int warp = thread_id / 32;
        const int lane = thread_id % 32;
        this->row = row + warp * 4 + lane / 8;
        const int thread_column = column + (lane % 8) * 8;
        pointer = source + this->row * stride + thread_column;
#pragma unroll
        for (int index = 0; index < 4; ++index)
        {
            valid_columns |= static_cast<unsigned>(thread_column + index * 64 < columns)
                             << index;
        }
    }

    __device__ const void *get(int index) const { return pointer + index * 64; }
    __device__ bool valid(int index) const
    {
        return row < extent_rows && ((valid_columns >> index) & 1U);
    }
    __device__ void add_tile_offset(int row_tiles, int column_tiles)
    {
        const int row_offset = row_tiles * 32;
        const int column_offset = column_tiles * 256;
        pointer += row_offset * leading_dimension + column_offset;
        row += row_offset;
    }
};

__global__ void initialize_matrix(__half *matrix, std::size_t count)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        matrix[index] = __float2half_rn(
            static_cast<float>(static_cast<int>(index % 17) - 8) * 0.01f);
    }
}

__global__ void direct_gemm_kernel(KernelParams params)
{

    extern __shared__ int shared_storage_base[];
    auto *operand_a = reinterpret_cast<__half *>(shared_storage_base);
    __half *operand_b = operand_a + SHARED_A_ELEMENTS;

    const int tile_offset_m = static_cast<int>(blockIdx.x) * MMA_TILE_M;
    const int tile_offset_n = static_cast<int>(blockIdx.y) * MMA_TILE_N;
    int gemm_k_iterations = (params.inner + MMA_TILE_K - 1) / MMA_TILE_K;
    const int thread_id = threadIdx.x;

    GlobalTileLoaderA iterator_a(params.matrix_a, params.stride_a, params.rows,
                                 params.inner, thread_id, tile_offset_m, 0);
    GlobalTileLoaderB iterator_b(params.matrix_b, params.stride_b, params.inner,
                                 params.columns, thread_id, 0, tile_offset_n);

    const int warp_id = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const int lane_id = thread_id % 32;

    const int warp_index_mn = warp_id % (WARP_COUNT_M * WARP_COUNT_N);
    const int warp_index_k = warp_id / (WARP_COUNT_M * WARP_COUNT_N);
    const int warp_index_m = warp_index_mn % WARP_COUNT_M;
    const int warp_index_n = warp_index_mn / WARP_COUNT_M;

    WarpTileLoaderA warp_tile_loader_a(operand_a, SHARED_A_STRIDE, lane_id);
    WarpTileLoaderB warp_tile_loader_b(operand_b, SHARED_B_STRIDE, lane_id);
    warp_tile_loader_a.add_tile_offset(warp_index_m,
                                       WARP_GEMM_ITERATIONS * warp_index_k);
    warp_tile_loader_b.add_tile_offset(WARP_GEMM_ITERATIONS * warp_index_k,
                                       warp_index_n);

    int shared_memory_write_stage = 0;
    int shared_memory_read_stage = 0;
    float accumulators[128] = {};

    SharedMemoryStoreA smem_iterator_a(operand_a, SHARED_A_STRIDE, thread_id);
    SharedMemoryStoreB smem_iterator_b(operand_b, SHARED_B_STRIDE, thread_id);

// Inline the multistage prologue: populate the first kStages - 1 shared-
// memory stages directly with cp.async operations.
#pragma unroll
    for (int stage = 0; stage < MMA_STAGES - 1; ++stage, --gemm_k_iterations)
    {
        smem_iterator_a.set_iteration_index(0);

#pragma unroll
        for (int access = 0; access < ASYNC_COPY_ITERATIONS_A; ++access)
        {
            void *destination = smem_iterator_a.get();
            async_copy_zfill_16(destination, iterator_a.get(access),
                                iterator_a.valid(access));
            smem_iterator_a.advance();
        }

        smem_iterator_b.set_iteration_index(0);

#pragma unroll
        for (int access = 0; access < ASYNC_COPY_ITERATIONS_B; ++access)
        {
            void *destination = smem_iterator_b.get();
            async_copy_zfill_16(destination, iterator_b.get(access),
                                iterator_b.valid(access));
            smem_iterator_b.advance();
        }

        smem_iterator_a.add_tile_offset(0, 1);
        smem_iterator_b.add_tile_offset(1, 0);
        iterator_a.add_tile_offset(0, 1);
        iterator_b.add_tile_offset(1, 0);

        ++shared_memory_write_stage;

        commit_async_copy_group();
    }

    wait_for_async_copy_groups<MMA_STAGES - 2>();
    __syncthreads();

    alignas(4) __half warp_loaded_fragments_a[2][32];
    alignas(4) __half warp_loaded_fragments_b[2][32];

    warp_tile_loader_a.set_kgroup_index(0);
    warp_tile_loader_a.load(warp_loaded_fragments_a[0]);
    warp_tile_loader_a.advance();
    warp_tile_loader_b.load(warp_loaded_fragments_b[0]);
    warp_tile_loader_b.advance();

#pragma unroll 1
    for (; gemm_k_iterations > (-MMA_STAGES + 1);)
    {
        #pragma unroll
        for (int warp_kgroup = 0; warp_kgroup < WARP_GEMM_ITERATIONS; ++warp_kgroup)
        {
            // Consume the current code
            const int next_warp_slot = (warp_kgroup + 1) % 2;
            warp_tile_loader_a.set_kgroup_index((warp_kgroup + 1) %
                                                WARP_GEMM_ITERATIONS);
            warp_tile_loader_a.load(warp_loaded_fragments_a[next_warp_slot]);
            warp_tile_loader_a.advance();
            warp_tile_loader_b.load(warp_loaded_fragments_b[next_warp_slot]);
            warp_tile_loader_b.advance();

            const int current_warp_slot = warp_kgroup % 2;
            const __half(*mma_a)[8] = reinterpret_cast<const __half(*)[8]>(
                warp_loaded_fragments_a[current_warp_slot]);
            const __half(*mma_b)[4] = reinterpret_cast<const __half(*)[4]>(
                warp_loaded_fragments_b[current_warp_slot]);
            float(*mma_c)[4] = reinterpret_cast<float(*)[4]>(accumulators);
                
            // Concrete execution logic
            #pragma unroll
            for (int n = 0; n < WARP_MMA_ITERATIONS_N; ++n)
            {
                #pragma unroll
                for (int m = 0; m < WARP_MMA_ITERATIONS_M; ++m)
                {
                    const int m_serpentine =
                        (n % 2) ? WARP_MMA_ITERATIONS_M - 1 - m : m;
                    mma_sync_m16n8k16(mma_c[m_serpentine + n * WARP_MMA_ITERATIONS_M],
                                      mma_a[m_serpentine], mma_b[n],
                                      mma_c[m_serpentine + n * WARP_MMA_ITERATIONS_M]);
                }
            }

            // Load the next part in the shared memory
            if (warp_kgroup == 0)
            {
                #pragma unroll
                for (int copy_group = 0; copy_group < WARP_GEMM_ITERATIONS;
                     ++copy_group)
                {
                    const int group_start_a = copy_group * ACCESSES_PER_GROUP_A;
                    smem_iterator_a.set_iteration_index(group_start_a);
                    #pragma unroll
                    for (int copy = 0; copy < ACCESSES_PER_GROUP_A; ++copy)
                    {

                        void *destination = smem_iterator_a.get();
                        const int access = group_start_a + copy;
                        async_copy_zfill_16(destination, iterator_a.get(access),
                                            iterator_a.valid(access));
                        smem_iterator_a.advance();
                    }

                    const int group_start_b = copy_group * ACCESSES_PER_GROUP_B;
                    smem_iterator_b.set_iteration_index(group_start_b);

                    #pragma unroll
                    for (int copy = 0; copy < ACCESSES_PER_GROUP_B; ++copy)
                    {

                        void *destination = smem_iterator_b.get();
                        const int access = group_start_b + copy;
                        async_copy_zfill_16(destination, iterator_b.get(access),
                                            iterator_b.valid(access));
                        smem_iterator_b.advance();
                    }
                }

                commit_async_copy_group();
                wait_for_async_copy_groups<MMA_STAGES - 2>();
                __syncthreads();

                iterator_a.add_tile_offset(0, 1);
                iterator_b.add_tile_offset(1, 0);
                smem_iterator_a.add_tile_offset(0, 1);
                smem_iterator_b.add_tile_offset(1, 0);
                ++shared_memory_write_stage;
                if (shared_memory_write_stage == MMA_STAGES)
                {
                    smem_iterator_a.add_tile_offset(0, -MMA_STAGES);
                    smem_iterator_b.add_tile_offset(-MMA_STAGES, 0);
                    shared_memory_write_stage = 0;
                }

                ++shared_memory_read_stage;
                if (shared_memory_read_stage == MMA_STAGES)
                {
                    constexpr int read_stage_offset = MMA_STAGES * WARP_GEMM_ITERATIONS;
                    warp_tile_loader_a.add_tile_offset(0, -read_stage_offset);
                    warp_tile_loader_b.add_tile_offset(-read_stage_offset, 0);
                    shared_memory_read_stage = 0;
                }

                --gemm_k_iterations;
            }
        }
    }

    commit_async_copy_group();
    wait_for_all_async_copies();
    __syncthreads();

    auto *output = params.matrix_d;
    const int output_stride = params.stride_d;
    const int output_warp = thread_id / 32;
    const int output_row_base = static_cast<int>(blockIdx.x) * MMA_TILE_M +
                                (output_warp / 4) * 64 + (output_warp % 4) * 2 +
                                lane_id / 16;
    const int output_column_base =
        static_cast<int>(blockIdx.y) * MMA_TILE_N + (lane_id % 16) * 8;

    const int warp_k = warp_id / (EPILOGUE_WARP_COUNT_M * EPILOGUE_WARP_COUNT_N);
    const int warp_mn = warp_id % (EPILOGUE_WARP_COUNT_M * EPILOGUE_WARP_COUNT_N);
    const int warp_m = warp_mn % EPILOGUE_WARP_COUNT_M;
    const int warp_n = warp_mn / EPILOGUE_WARP_COUNT_M;
    const auto *accumulator_accesses = reinterpret_cast<const float2 *>(accumulators);
    auto *shared_accumulators = reinterpret_cast<float2 *>(shared_storage_base);
    constexpr int shared_stride = MMA_TILE_N / 2;
    const int warp_tile_row = (warp_k * EPILOGUE_WARP_COUNT_M + warp_m) * 8;
    const int warp_tile_column = warp_n * 32;
    const int lane_row = lane_id / 4;
    const auto *shared_loads = reinterpret_cast<const float4 *>(shared_storage_base);
    constexpr int shared_load_stride = MMA_TILE_N / 4;
    const int load_warp = thread_id / 32;
    const int load_lane = thread_id % 32;
    const int load_row = load_warp * 2 + load_lane / 16;
    const int load_column_index = (load_lane % 16) * 2;
    const int load_bank = (load_column_index / 8) % 2;

#pragma unroll
    for (int iteration = 0; iteration < EPILOGUE_ITERATIONS; ++iteration)
    {
        __syncthreads();

#pragma unroll
        for (int column = 0; column < 8; ++column)
        {
            const int pointer_bank = column / 4;
            const int lane_column =
                (lane_id % 2) + (((lane_id / 2 + pointer_bank) % 2) * 2);

            shared_accumulators[(warp_tile_row + lane_row) * shared_stride +
                                warp_tile_column + lane_column + column * 4] =
                accumulator_accesses[iteration + column * 8];
        }

        __syncthreads();

        float4 aligned_fragment[4];

#pragma unroll
        for (int column = 0; column < 2; ++column)
        {
#pragma unroll
            for (int vector = 0; vector < 2; ++vector)
            {
                const int vector_column = load_column_index + (load_bank + vector) % 2;
                aligned_fragment[column * 2 + vector] =
                    shared_loads[load_row * shared_load_stride + vector_column +
                                 column * 32];
            }
        }

        const auto *accumulated = reinterpret_cast<const float *>(aligned_fragment);
        alignas(16) __half output_fragment[16];

#pragma unroll
        for (int element = 0; element < 8; ++element)
        {
            output_fragment[element] = __float2half_rn(accumulated[element]);
            output_fragment[element + 8] = __float2half_rn(accumulated[element + 8]);
        }

        const int output_row = output_row_base + iteration * 8;
        if (output_row < params.rows)
        {
            const auto *output_values =
                reinterpret_cast<const uint4 *>(output_fragment);
            auto *output_vectors = reinterpret_cast<uint4 *>(
                output + output_row * output_stride + output_column_base);
            output_vectors[0] = output_values[0];
            output_vectors[16] = output_values[1];
        }
    }
}

int main()
{
    constexpr std::size_t elements =
        static_cast<std::size_t>(MATRIX_SIZE) * MATRIX_SIZE;
    constexpr std::size_t bytes = elements * sizeof(__half);
    __half *matrix_a = nullptr;
    __half *matrix_b = nullptr;
    __half *matrix_c = nullptr;
    __half *matrix_reference = nullptr;
    CUDA_CHECK(cudaMalloc(&matrix_a, bytes));
    CUDA_CHECK(cudaMalloc(&matrix_b, bytes));
    CUDA_CHECK(cudaMalloc(&matrix_c, bytes));
    CUDA_CHECK(cudaMalloc(&matrix_reference, bytes));

    initialize_matrix<<<(elements + 255) / 256, 256>>>(matrix_a, elements);
    initialize_matrix<<<(elements + 255) / 256, 256>>>(matrix_b, elements);
    CUDA_CHECK(cudaGetLastError());

    const KernelParams params{matrix_a,    matrix_b,    matrix_c,
                              MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE,
                              MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE};
    const dim3 grid((MATRIX_SIZE + MMA_TILE_M - 1) / MMA_TILE_M,
                    (MATRIX_SIZE + MMA_TILE_N - 1) / MMA_TILE_N, 1);
    const dim3 block(THREAD_COUNT);
    CUDA_CHECK(cudaFuncSetAttribute(direct_gemm_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    SHARED_MEMORY_BYTES));

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration)
    {
        direct_gemm_kernel<<<grid, block, SHARED_MEMORY_BYTES>>>(params);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
    {
        direct_gemm_kernel<<<grid, block, SHARED_MEMORY_BYTES>>>(params);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    milliseconds /= BENCHMARK_ITERATIONS;

    cublasHandle_t cublas_handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, MATRIX_SIZE,
                              MATRIX_SIZE, MATRIX_SIZE, &alpha, matrix_b, CUDA_R_16F,
                              MATRIX_SIZE, matrix_a, CUDA_R_16F, MATRIX_SIZE, &beta,
                              matrix_reference, CUDA_R_16F, MATRIX_SIZE,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> output(elements);
    std::vector<__half> reference(elements);
    CUDA_CHECK(cudaMemcpy(output.data(), matrix_c, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(reference.data(), matrix_reference, bytes, cudaMemcpyDeviceToHost));
    float max_absolute_error = 0.0f;
    for (std::size_t index = 0; index < elements; ++index)
    {
        max_absolute_error =
            std::max(max_absolute_error, std::abs(__half2float(output[index]) -
                                                  __half2float(reference[index])));
    }
    if (max_absolute_error > 0.01f)
    {
        std::cerr << "Validation failed: max_absolute_error=" << max_absolute_error
                  << '\n';
        return EXIT_FAILURE;
    }

    const double tflops = 2.0 * MATRIX_SIZE * MATRIX_SIZE *
                          static_cast<double>(MATRIX_SIZE) / milliseconds / 1.0e9;
    std::cout << std::fixed << std::setprecision(3)
              << "kernel=direct_cuda_body size=4096 time_ms=" << milliseconds
              << " TFLOP/s=" << tflops << " max_abs_error=" << max_absolute_error
              << '\n';

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(matrix_a));
    CUDA_CHECK(cudaFree(matrix_b));
    CUDA_CHECK(cudaFree(matrix_c));
    CUDA_CHECK(cudaFree(matrix_reference));
    return EXIT_SUCCESS;
}
