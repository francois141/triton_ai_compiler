#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)      \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";   \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int WARMUP_ITERATIONS = 5;
constexpr int BENCHMARK_ITERATIONS = 10;
constexpr int WMMA_TILE_SIZE = 16;
constexpr int WARP_SIZE = 32;
constexpr int HALF_VALUES_PER_VECTOR = sizeof(uint4) / sizeof(__half);
constexpr int SHARED_MEMORY_PADDING_VALUES = HALF_VALUES_PER_VECTOR;
constexpr int MAX_STATIC_SHARED_MEMORY_BYTES = 48 * 1024;
constexpr int VERIFICATION_SAMPLES = 8;
constexpr double OPERATIONS_PER_MATMUL =
    2.0 * 4096 * 4096 * 4096;

__host__ __device__ float input_value(std::size_t index)
{
    return static_cast<float>(static_cast<int>(index % 17) - 8) * 0.01f;
}

__global__ void initialize_matrix(__half* matrix, std::size_t element_count)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < element_count) {
        matrix[index] = __float2half_rn(input_value(index));
    }
}

bool verify_matmul(const __half* matrix_c)
{
    constexpr int sample_rows[VERIFICATION_SAMPLES] = {
        0, 1, 127, 1023, 2048, 3071, 4094, 4095};
    constexpr int sample_columns[VERIFICATION_SAMPLES] = {
        0, 17, 511, 1537, 2048, 3001, 4078, 4095};
    constexpr float tolerance = 0.015625f;

    for (int sample = 0; sample < VERIFICATION_SAMPLES; ++sample) {
        const int row = sample_rows[sample];
        const int column = sample_columns[sample];
        float expected = 0.0f;

        for (int k = 0; k < 4096; ++k) {
            const float a = __half2float(__float2half_rn(
                input_value(static_cast<std::size_t>(row) * 4096 + k)));
            const float b = __half2float(__float2half_rn(
                input_value(static_cast<std::size_t>(k) * 4096 + column)));
            expected = fmaf(a, b, expected);
        }

        __half actual_half{};
        CUDA_CHECK(cudaMemcpy(
            &actual_half,
            matrix_c + static_cast<std::size_t>(row) * 4096 + column,
            sizeof(actual_half),
            cudaMemcpyDeviceToHost));

        const float actual = __half2float(actual_half);
        const float expected_half = __half2float(__float2half_rn(expected));
        if (std::fabs(actual - expected_half) > tolerance) {
            std::cerr << "Verification failed at (" << row << ", " << column
                      << "): expected " << expected_half
                      << ", got " << actual << '\n';
            return false;
        }
    }

    return true;
}

template <int WARP_TILES_M, int WARP_TILES_N, int TILE_K>
__global__ void matmul_fp16(
    const __half* __restrict__ matrix_a,
    const __half* __restrict__ matrix_b,
    __half* __restrict__ matrix_c)
{
    namespace wmma = nvcuda::wmma;

    constexpr int BLOCK_TILE_M = WARP_TILES_M * WMMA_TILE_SIZE;
    constexpr int BLOCK_TILE_N = WARP_TILES_N * WMMA_TILE_SIZE;

    constexpr int THREADS_PER_BLOCK = WARP_SIZE;

    constexpr int A_VECTORS_PER_ROW = TILE_K / HALF_VALUES_PER_VECTOR;
    constexpr int B_VECTORS_PER_ROW = BLOCK_TILE_N / HALF_VALUES_PER_VECTOR;

    constexpr int SHARED_MEMORY_BYTES =
        sizeof(__half) * (BLOCK_TILE_M * TILE_K + TILE_K * BLOCK_TILE_N) +
        sizeof(float) * BLOCK_TILE_M * BLOCK_TILE_N;

    constexpr int PADDED_SHARED_MEMORY_BYTES =
        SHARED_MEMORY_BYTES +
        sizeof(__half) * SHARED_MEMORY_PADDING_VALUES *
            (BLOCK_TILE_M + TILE_K);

    constexpr int SHARED_MEMORY_PADDING =
        PADDED_SHARED_MEMORY_BYTES <= MAX_STATIC_SHARED_MEMORY_BYTES
            ? SHARED_MEMORY_PADDING_VALUES
            : 0;
            
    constexpr int A_SHARED_STRIDE = TILE_K + SHARED_MEMORY_PADDING;
    constexpr int B_SHARED_STRIDE = BLOCK_TILE_N + SHARED_MEMORY_PADDING;

    static_assert(TILE_K % WMMA_TILE_SIZE == 0);
    static_assert(TILE_K % HALF_VALUES_PER_VECTOR == 0);
    static_assert(BLOCK_TILE_N % HALF_VALUES_PER_VECTOR == 0);
    static_assert(A_SHARED_STRIDE % HALF_VALUES_PER_VECTOR == 0);
    static_assert(B_SHARED_STRIDE % HALF_VALUES_PER_VECTOR == 0);
    static_assert(WARP_TILES_M > 0);
    static_assert(WARP_TILES_N > 0);

    __shared__ __half tile_a[BLOCK_TILE_M][A_SHARED_STRIDE];
    __shared__ __half tile_b[TILE_K][B_SHARED_STRIDE];
    __shared__ float tile_c[BLOCK_TILE_M][BLOCK_TILE_N];

    const int thread_id = threadIdx.x;
    const int block_row = blockIdx.y * BLOCK_TILE_M;
    const int block_col = blockIdx.x * BLOCK_TILE_N;

    wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major>
        a_fragments[WARP_TILES_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>
        b_fragments[WARP_TILES_N];
    wmma::fragment<wmma::accumulator, 16,16,16, float>
        c_fragments[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; ++warp_tile_m) {
        #pragma unroll
        for (int warp_tile_n = 0;
             warp_tile_n < WARP_TILES_N;
             ++warp_tile_n) {
            wmma::fill_fragment(
                c_fragments[warp_tile_m][warp_tile_n],
                0.0f);
        }
    }

    for (int k_base = 0; k_base < 4096; k_base += TILE_K) {
        for (int vector_index = thread_id;
             vector_index < BLOCK_TILE_M * A_VECTORS_PER_ROW;
             vector_index += THREADS_PER_BLOCK) {
            const int tile_row = vector_index / A_VECTORS_PER_ROW;
            const int tile_col =
                (vector_index % A_VECTORS_PER_ROW) * HALF_VALUES_PER_VECTOR;

            const uint4 values = *reinterpret_cast<const uint4*>(
                &matrix_a[(block_row + tile_row) * 4096 +
                          k_base + tile_col]);
            *reinterpret_cast<uint4*>(&tile_a[tile_row][tile_col]) = values;
        }

        for (int vector_index = thread_id;
             vector_index < TILE_K * B_VECTORS_PER_ROW;
             vector_index += THREADS_PER_BLOCK) {
            const int tile_row = vector_index / B_VECTORS_PER_ROW;
            const int tile_col =
                (vector_index % B_VECTORS_PER_ROW) * HALF_VALUES_PER_VECTOR;
            const uint4 values = *reinterpret_cast<const uint4*>(
                &matrix_b[(k_base + tile_row) * 4096 +
                          block_col + tile_col]);
            *reinterpret_cast<uint4*>(&tile_b[tile_row][tile_col]) = values;
        }
        __syncthreads();

        #pragma unroll
        for (int k_offset = 0; k_offset < TILE_K;
             k_offset += WMMA_TILE_SIZE) {
            #pragma unroll
            for (int warp_tile_m = 0;
                 warp_tile_m < WARP_TILES_M;
                 ++warp_tile_m) {
                wmma::load_matrix_sync(
                    a_fragments[warp_tile_m],
                    &tile_a[warp_tile_m * WMMA_TILE_SIZE][k_offset],
                    A_SHARED_STRIDE);
            }

            #pragma unroll
            for (int warp_tile_n = 0;
                 warp_tile_n < WARP_TILES_N;
                 ++warp_tile_n) {
                wmma::load_matrix_sync(
                    b_fragments[warp_tile_n],
                    &tile_b[k_offset][warp_tile_n * WMMA_TILE_SIZE],
                    B_SHARED_STRIDE);
            }

            #pragma unroll
            for (int warp_tile_m = 0;
                 warp_tile_m < WARP_TILES_M;
                 ++warp_tile_m) {
                #pragma unroll
                for (int warp_tile_n = 0;
                     warp_tile_n < WARP_TILES_N;
                     ++warp_tile_n) {
                    wmma::mma_sync(
                        c_fragments[warp_tile_m][warp_tile_n],
                        a_fragments[warp_tile_m],
                        b_fragments[warp_tile_n],
                        c_fragments[warp_tile_m][warp_tile_n]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int warp_tile_m = 0; warp_tile_m < WARP_TILES_M; ++warp_tile_m) {
        #pragma unroll
        for (int warp_tile_n = 0;
             warp_tile_n < WARP_TILES_N;
             ++warp_tile_n) {
            wmma::store_matrix_sync(
                &tile_c[warp_tile_m * WMMA_TILE_SIZE]
                       [warp_tile_n * WMMA_TILE_SIZE],
                c_fragments[warp_tile_m][warp_tile_n],
                BLOCK_TILE_N,
                wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int element = thread_id;
         element < BLOCK_TILE_M * BLOCK_TILE_N;
         element += THREADS_PER_BLOCK) {
        const int tile_row = element / BLOCK_TILE_N;
        const int tile_col = element % BLOCK_TILE_N;
        matrix_c[(block_row + tile_row) * 4096 + block_col + tile_col] =
            __float2half_rn(tile_c[tile_row][tile_col]);
    }
}

template <int WARP_TILES_M, int WARP_TILES_N, int TILE_K>
void benchmark_configuration(
    const __half* matrix_a,
    const __half* matrix_b,
    __half* matrix_c,
    cudaEvent_t start,
    cudaEvent_t stop)
{
    constexpr int BLOCK_TILE_M = WARP_TILES_M * WMMA_TILE_SIZE;
    constexpr int BLOCK_TILE_N = WARP_TILES_N * WMMA_TILE_SIZE;
    constexpr int THREADS_PER_BLOCK = WARP_SIZE;

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(
        4096 / BLOCK_TILE_N,
        4096 / BLOCK_TILE_M);

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        matmul_fp16<WARP_TILES_M, WARP_TILES_N, TILE_K><<<grid, block>>>(
            matrix_a,
            matrix_b,
            matrix_c);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration) {
        matmul_fp16<WARP_TILES_M, WARP_TILES_N, TILE_K><<<grid, block>>>(
            matrix_a,
            matrix_b,
            matrix_c);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_milliseconds,
        start,
        stop));

    const double average_milliseconds =
        elapsed_milliseconds / BENCHMARK_ITERATIONS;
    const double tflops = OPERATIONS_PER_MATMUL / average_milliseconds / 1.0e9;

    if (!verify_matmul(matrix_c)) {
        std::exit(EXIT_FAILURE);
    }

    std::cout << std::setw(5) << WARP_TILES_M
              << std::setw(5) << WARP_TILES_N
              << std::setw(5) << BLOCK_TILE_M
              << std::setw(5) << BLOCK_TILE_N
              << std::setw(5) << TILE_K
              << std::setw(8) << THREADS_PER_BLOCK
              << std::setw(12) << std::setprecision(3) << std::fixed
              << average_milliseconds
              << std::setw(12) << tflops
              << "  PASS\n";
}

int main()
{
    constexpr std::size_t MATRIX_BYTES =
        static_cast<std::size_t>(4096) * 4096 * sizeof(__half);

    __half* matrix_a = nullptr;
    __half* matrix_b = nullptr;
    __half* matrix_c = nullptr;
    CUDA_CHECK(cudaMalloc(&matrix_a, MATRIX_BYTES));
    CUDA_CHECK(cudaMalloc(&matrix_b, MATRIX_BYTES));
    CUDA_CHECK(cudaMalloc(&matrix_c, MATRIX_BYTES));

    constexpr int INITIALIZATION_THREADS = 256;
    const int initialization_blocks = static_cast<int>(
        (static_cast<std::size_t>(4096) * 4096 +
         INITIALIZATION_THREADS - 1) /
        INITIALIZATION_THREADS);
    initialize_matrix<<<initialization_blocks, INITIALIZATION_THREADS>>>(
        matrix_a,
        static_cast<std::size_t>(4096) * 4096);
    initialize_matrix<<<initialization_blocks, INITIALIZATION_THREADS>>>(
        matrix_b,
        static_cast<std::size_t>(4096) * 4096);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "Matrix: 4096 x 4096\n";
    std::cout << "Work per matmul: " << std::fixed << std::setprecision(3)
              << OPERATIONS_PER_MATMUL / 1.0e12 << " TFLOP\n\n";
    std::cout << "  WM   WN   BM   BN   BK Threads      ms     TFLOP/s  Verification\n";

    benchmark_configuration<2, 4, 32>(
        matrix_a, matrix_b, matrix_c, start, stop);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(matrix_a));
    CUDA_CHECK(cudaFree(matrix_b));
    CUDA_CHECK(cudaFree(matrix_c));
    return 0;
}
