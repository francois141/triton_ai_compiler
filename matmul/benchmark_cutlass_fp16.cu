#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        const cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)      \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";     \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

#define CUTLASS_CHECK(call)                                                \
    do {                                                                   \
        const cutlass::Status status = (call);                             \
        if (status != cutlass::Status::kSuccess) {                         \
            std::cerr << "CUTLASS error: "                                \
                      << cutlassGetStatusString(status) << '\n';           \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

constexpr int DEFAULT_SIZE = 4096;
constexpr int WARMUP_ITERATIONS = 20;
constexpr int DEFAULT_BENCHMARK_ITERATIONS = 100;
constexpr int THREADS_PER_BLOCK = 256;

__global__ void initialize_matrix(
    cutlass::half_t* matrix,
    std::size_t element_count)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < element_count) {
        const float value =
            static_cast<float>(static_cast<int>(index % 17) - 8) * 0.01f;
        matrix[index] = cutlass::half_t(value);
    }
}

int main(int argc, char* argv[])
{
    const int size = argc > 1 ? std::atoi(argv[1]) : DEFAULT_SIZE;
    const int iterations =
        argc > 2 ? std::atoi(argv[2]) : DEFAULT_BENCHMARK_ITERATIONS;
    if (size <= 0 || iterations <= 0) {
        std::cerr << "Size and iterations must be positive.\n";
        return EXIT_FAILURE;
    }

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,
        8,
        float,
        float>;
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3>;

    const std::size_t elements = static_cast<std::size_t>(size) * size;
    const std::size_t bytes = elements * sizeof(cutlass::half_t);
    cutlass::half_t* matrix_a = nullptr;
    cutlass::half_t* matrix_b = nullptr;
    cutlass::half_t* matrix_c = nullptr;
    CUDA_CHECK(cudaMalloc(&matrix_a, bytes));
    CUDA_CHECK(cudaMalloc(&matrix_b, bytes));
    CUDA_CHECK(cudaMalloc(&matrix_c, bytes));

    const int blocks = static_cast<int>(
        (elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    initialize_matrix<<<blocks, THREADS_PER_BLOCK>>>(matrix_a, elements);
    initialize_matrix<<<blocks, THREADS_PER_BLOCK>>>(matrix_b, elements);
    CUDA_CHECK(cudaGetLastError());

    Gemm gemm;
    const typename Gemm::Arguments arguments(
        {size, size, size},
        {matrix_a, size},
        {matrix_b, size},
        {matrix_c, size},
        {matrix_c, size},
        {1.0f, 0.0f});
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments));

    for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
        CUTLASS_CHECK(gemm());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        CUTLASS_CHECK(gemm());
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));
    const double milliseconds = total_milliseconds / iterations;
    const double tflops = 2.0 * size * size * static_cast<double>(size) /
                          milliseconds / 1.0e9;
    std::cout << std::fixed << std::setprecision(3)
              << "kernel=cutlass_fp16_tensorop_fp32_accum size=" << size
              << " time_ms=" << milliseconds << " TFLOP/s=" << tflops
              << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(matrix_a));
    CUDA_CHECK(cudaFree(matrix_b));
    CUDA_CHECK(cudaFree(matrix_c));
    return EXIT_SUCCESS;
}


/*

/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_89 \
  -I/tmp/cutlass/include matmul/benchmark_cutlass_fp16.cu \
  -o matmul/benchmark_cutlass_fp16

./matmul/benchmark_cutlass_fp16 4096 100
*/
