// benchmark_cublas_fp16.cu

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t status_ = (call);                                        \
        if (status_ != cudaSuccess) {                                        \
            std::cerr << "CUDA error: " << cudaGetErrorString(status_)       \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t status_ = (call);                                     \
        if (status_ != CUBLAS_STATUS_SUCCESS) {                              \
            std::cerr << "cuBLAS error " << static_cast<int>(status_)        \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void initialize_half(
    __half* data,
    std::size_t count,
    float scale)
{
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < count) {
        // Valeurs déterministes suffisamment petites pour éviter les overflow.
        const int value = static_cast<int>(index % 17) - 8;
        data[index] = __float2half(static_cast<float>(value) * scale);
    }
}

double percentile(std::vector<float> values, double p)
{
    if (values.empty()) {
        throw std::runtime_error("Empty timing vector");
    }

    std::sort(values.begin(), values.end());

    const double position = p * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);

    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

int main()
{
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;

    constexpr int warmup_iterations = 100;
    constexpr int measured_samples = 200;
    constexpr int gemms_per_sample = 20;

    const std::size_t elements_a =
        static_cast<std::size_t>(M) * K;
    const std::size_t elements_b =
        static_cast<std::size_t>(K) * N;
    const std::size_t elements_c =
        static_cast<std::size_t>(M) * N;

    __half* d_a = nullptr;
    __half* d_b = nullptr;
    __half* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, elements_a * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_b, elements_b * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_c, elements_c * sizeof(__half)));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &stream,
        cudaStreamNonBlocking));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // Autorise explicitement les Tensor Cores.
    CUBLAS_CHECK(cublasSetMathMode(
        handle,
        CUBLAS_TENSOR_OP_MATH));

    constexpr int threads = 256;

    const int blocks_a = static_cast<int>(
        (elements_a + threads - 1) / threads);
    const int blocks_b = static_cast<int>(
        (elements_b + threads - 1) / threads);

    initialize_half<<<blocks_a, threads, 0, stream>>>(
        d_a,
        elements_a,
        0.01f);

    initialize_half<<<blocks_b, threads, 0, stream>>>(
        d_b,
        elements_b,
        0.01f);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemsetAsync(
        d_c,
        0,
        elements_c * sizeof(__half),
        stream));

    /*
       cuBLAS utilise par défaut un stockage column-major.

       A : M x K, lda = M
       B : K x N, ldb = K
       C : M x N, ldc = M

       Calcul :
           C = alpha * A * B + beta * C
    */

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto launch_gemm = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            handle,

            CUBLAS_OP_N,
            CUBLAS_OP_N,

            M,
            N,
            K,

            &alpha,

            d_a,
            CUDA_R_16F,
            M,

            d_b,
            CUDA_R_16F,
            K,

            &beta,

            d_c,
            CUDA_R_16F,
            M,

            CUBLAS_COMPUTE_32F,

            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Première exécution : initialisation interne éventuelle de cuBLAS.
    launch_gemm();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warm-up thermique et stabilisation du chemin d'exécution.
    for (int i = 0; i < warmup_iterations; ++i) {
        launch_gemm();
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> timings_ms;
    timings_ms.reserve(measured_samples);

    for (int sample = 0; sample < measured_samples; ++sample) {
        CUDA_CHECK(cudaEventRecord(start, stream));

        for (int repetition = 0;
             repetition < gemms_per_sample;
             ++repetition) {
            launch_gemm();
        }

        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float batch_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &batch_ms,
            start,
            stop));

        timings_ms.push_back(
            batch_ms /
            static_cast<float>(gemms_per_sample));
    }

    const double p20_ms = percentile(timings_ms, 0.20);
    const double p50_ms = percentile(timings_ms, 0.50);
    const double p80_ms = percentile(timings_ms, 0.80);
    const double p95_ms = percentile(timings_ms, 0.95);

    const double average_ms =
        std::accumulate(
            timings_ms.begin(),
            timings_ms.end(),
            0.0) /
        static_cast<double>(timings_ms.size());

    const double operations =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);

    auto calculate_tflops = [&](double milliseconds) {
        // operations / (ms * 1e-3) / 1e12
        // équivaut à operations / ms / 1e9.
        return operations / milliseconds / 1.0e9;
    };

    cudaDeviceProp properties{};
    int device = 0;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(
        &properties,
        device));

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "GPU: " << properties.name << '\n';
    std::cout << "GEMM: "
              << M << " x "
              << N << " x "
              << K << '\n';

    std::cout << "Input: FP16\n";
    std::cout << "Accumulation: FP32\n";
    std::cout << "Output: FP16\n";
    std::cout << "Samples: " << measured_samples << '\n';
    std::cout << "GEMMs per sample: "
              << gemms_per_sample << "\n\n";

    std::cout << "p20: " << p20_ms << " ms"
              << " | " << calculate_tflops(p20_ms)
              << " TFLOP/s\n";

    std::cout << "p50: " << p50_ms << " ms"
              << " | " << calculate_tflops(p50_ms)
              << " TFLOP/s\n";

    std::cout << "p80: " << p80_ms << " ms"
              << " | " << calculate_tflops(p80_ms)
              << " TFLOP/s\n";

    std::cout << "p95: " << p95_ms << " ms"
              << " | " << calculate_tflops(p95_ms)
              << " TFLOP/s\n";

    std::cout << "Mean: " << average_ms << " ms"
              << " | " << calculate_tflops(average_ms)
              << " TFLOP/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}

// /usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -arch=sm_89 -lcublas  matmul/cublas.cu -o matmul/cublas && ./matmul/cublas