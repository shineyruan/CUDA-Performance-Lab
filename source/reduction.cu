#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "memBenchmark.h"
#include "termcolor.hpp"

// NVTX Dir: C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt
#include <nvToolsExt.h>

#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

// Number of element to reduce
static const int n_elements = 32 * 1024 * 1024;

// Number of threads per block to use for all kernels
static const int threads = 256;

struct DIMS1D {
  int dimThreads;
  int dimBlocks;
};

#define CUDA(call)                                       \
  do {                                                   \
    cudaError_t e = (call);                              \
    if (e == cudaSuccess) break;                         \
    fprintf(stderr, __FILE__ ":%d: %s (%d)\n", __LINE__, \
            cudaGetErrorString(e), e);                   \
    exit(1);                                             \
  } while (0)

inline unsigned divup(unsigned n, unsigned div) { return (n + div - 1) / div; }

void printResults(double timeInMilliseconds, int iterations) {
  // print out the time required for the kernel to finish the transpose
  // operation
  double bandwidth =
      (iterations * 1e-9 * (double)(n_elements * sizeof(float))) /
      (timeInMilliseconds * 1e-3);
  std::cout << "Elapsed Time for " << iterations
            << " runs = " << round(timeInMilliseconds) << "ms" << std::endl;
  std::cout << termcolor::bold << termcolor::red << termcolor::on_white
            << "Bandwidth (GB/s) = " << std::setprecision(4) << bandwidth
            << termcolor::reset << std::endl;
  std::cout.clear();
}

// Check errors
bool postprocess(const float *ref, const float *res, int n) {
  bool passed = true;
  for (int i = 0; i < n; i++) {
    if (std::abs(res[i] - ref[i]) / n_elements > 1e-6) {
      std::cout.precision(6);
      std::cout << "ID: " << i << " \t Res: " << res[i] << " \t Ref: " << ref[i]
                << std::endl;
      std::cout << termcolor::blink << termcolor::white << termcolor::on_red
                << "*** FAILED ***" << termcolor::reset << std::endl;
      passed = false;
      break;
    }
  }
  if (passed)
    std::cout << termcolor::green << "Post process check passed!!"
              << termcolor::reset << std::endl;

  return passed;
}

static float reduce_cpu(const float *data, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) sum += data[i];
  return sum;
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 0: Interleaved Addressing
// d_idata : Device pointer to input
// d_odata : Device pointer to output
// n       : Number of elements to reduce
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_stage0(const float *d_idata, float *d_odata, int n) {
  // Dynamic allocation of shared memory - See kernel call in host code
  extern __shared__ float smem[];

  // Calculate 1D Index
  int idx = 0;

  // Copy input data to shared memory
  // Note: Use block index for shared memory
  // Also check for bounds

  // Where do we need to put __syncthreads()? Do we need it at all?

  // Reduce within block
  // Start from c = 1, upto block size, each time doubling the offset

  // Copy result of reduction to global memory
  // Which index of d_odata do we write to?
  // In which index of smem is the result stored?
  // Do we need another syncthreads before writing to global memory?
  // Use only one thread to write to global memory
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 1: Non-divergent Addressing
// d_idata : Device pointer to input
// d_odata : Device pointer to output
// n       : Number of elements to reduce
//
// The only difference between stage0 and stage1 is the reduction for loop
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_stage1(const float *d_idata, float *d_odata, int n) {
  // Allocate dynamic shared memory, Calculate 1D Index and Copy input data into
  // shared memory Exactly same as reduce_stage0

  extern __shared__ float smem[];

  int idx = 0;

  // This is the part that differes from reduce_stage0
  // Reduce within block with coalesced indexing pattern
  // Change the for-loop to use indexing that reduces warp-divergence

  // Copy result of reduction to global memory - Same as reduce_stage0
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 2: Warp Management without Bank Conflicts
// d_idata : Device pointer to input
// d_odata : Device pointer to output
// n       : Number of elements to reduce
//
// The only difference between stage1 and stage2 is the reduction for loop
// This time, we reduce start from blockDim.x and divide by 2 in each iteration
////////////////////////////////////////////////////////////////////////////////
__global__ void reduce_stage2(const float *d_idata, float *d_odata, int n) {
  // Allocate dynamic shared memory, Calculate 1D Index and Copy input data into
  // shared memory Exactly same as reduce_stage1

  extern __shared__ float smem[];

  int idx = 0;

  // This is the part that differes from reduce_stage1
  // Reduce within block with coalesced indexing pattern and avoid bank
  // conflicts Change the for-loop to use indexing that reduces warp-divergence
  // Start from blockDim.x / 2 and divide by 2 until we hit 1

  // Copy result of reduction to global memory - Same as reduce_stage1
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 3: Add During Load, Use tile to reduce number of blocks
// d_idata      : Device pointer to input
// d_odata      : Device pointer to output
// n            : Number of elements to reduce
// stage3_TILE  : Tiling factor
//
// In this kernel, we will add on load when copying data into shared memory
// The difference between stage3 and stage2 is how we load data into shared
// memory Each block does work of stage3_TILE * blockDim.x elements
////////////////////////////////////////////////////////////////////////////////
const int stage3_TILE = 2;
__global__ void reduce_stage3(const float *d_idata, float *d_odata, int n) {
  // Allocate dynamic shared memory
  extern __shared__ float smem[];

  // Calculate 1D index similar to stage2, but multiply by stage3_TILE
  int idx = 0;

  // Copy input data to shared memory. Add on load.

  // Reduce the block same as reduce_stage2

  // Copy result of reduction to global memory - Same as reduce_stage2
}

// warpReduce function for reduce_stag4 that reduces 2 warps into one value
__device__ void warpReduce(volatile float *smem, int tid) {
  // Write code for warp reduce here
  smem[tid] += smem[tid + 32];
  smem[tid] += smem[tid + 16];
  smem[tid] += smem[tid + 8];
  smem[tid] += smem[tid + 4];
  smem[tid] += smem[tid + 2];
  smem[tid] += smem[tid + 1];
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 4: Warp Loop Unrolling
// d_idata      : Device pointer to input
// d_odata      : Device pointer to output
// n            : Number of elements to reduce
// stage4_TILE  : Tiling factor - How does tuning this change performance?
//
// The reduce_stage4 kernel improves on reduce_stage3 by unrolling the block
// reduction by unrolling the loop that operates within a warp.
// Each block does work of stage4_TILE * blockDim.x elements
//
// This kernel also uses the warpReduce device function above
////////////////////////////////////////////////////////////////////////////////
const int stage4_TILE = 2;
__global__ void reduce_stage4(const float *d_idata, float *d_odata, int n) {
  // Allocate dynamic shared memory, Calculate 1D Index and
  // Copy input data and add on load into shared memory
  // Exactly same as reduce_stage3. Use stage4_TILE instead of stage3_TILE.

  extern __shared__ float smem[];

  int idx = 0;

  // Reduce within block with coalesced indexing pattern and avoid bank
  // conflicts Split the block reduction into 2 parts. Part 1 is the same as
  // reduce stage3, but only for c > 32

  // Part 2 then uses the warpReduce function to reduce the 2 warps
  // The reason we stop the previous loop at c > 32 is because
  // warpReduce can reduce 2 warps only 1 warp

  // Copy result of reduction to global memory - Same as reduce_stage3
}

////////////////////////////////////////////////////////////////////////////////
// Reduction Stage 5: Completely unrolled blocks using templates
// d_idata      : Device pointer to input
// d_odata      : Device pointer to output
// n            : Number of elements to reduce
// stage5_TILE  : Tiling factor - How does tuning this change performance?
//
// The reduce_stage5 kernel is the same as reduce_stage4 except part 1 of block
// reduction We simply unroll the entire for loop into individual statements
// wrapper by if conditions Why do we need to use templates? How do they improve
// performance? Each block does work of stage5_TILE * blockDim.x elements
//
// This kernel also uses the warpReduce device function above
////////////////////////////////////////////////////////////////////////////////
const int stage5_TILE = 2;
template <unsigned int blockSize>
__global__ void reduce_stage5(const float *d_idata, float *d_odata, int n) {
  // Allocate dynamic shared memory, Calculate 1D Index and
  // Copy input data and add on load into shared memory
  // Exactly same as reduce_stage4. Use stage5_TILE instead of stage4_TILE.
  // Use #pragma unroll around the load loop

  extern __shared__ float smem[];

  int idx = 0;

  // Store the threadIdx.x in a register
  int tid = threadIdx.x;

  // Reduce the block using the same part1 and part2 split that we used in
  // reduce_stage4 Except, here write explicit statements instead of the for
  // loops

  // Part 2 is the same as reduce_stage4

  // Copy result of reduction to global memory - Same as reduce_stage4
}

int main() {
  // Calculate bytes needed for input
  const unsigned bytes = n_elements * sizeof(float);

  // Allocate memory and initialize elements
  // Let's use pinned memory for host
  float *h_idata;
  CUDA(cudaMallocHost((void **)&h_idata, bytes));

  // Fill random values into the host array
  {
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_real_distribution<float> distribution(-1, 1);

    for (int i = 0; i < n_elements; i++) {
      h_idata[i] = distribution(generator);
    }
  }

  // Copy input data into device memory
  float *d_idata = NULL;
  CUDA(cudaMalloc((void **)&d_idata, bytes));
  CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

  // Compute Gold Standard using CPU
  const float gold_result = reduce_cpu(h_idata, n_elements);

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***CPU Reduce***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("CPU Reduce");

    float cpu_result = 0;

    int iterations = 10;

    // start the timer
    Timer hTimer;
    nvtxRangeId_t rangeBenchmark = nvtxRangeStart("CPU Reduce Benchmark");
    for (int k = 0; k < iterations; k++) {
      cpu_result = reduce_cpu(h_idata, n_elements);
    }
    nvtxRangeEnd(rangeBenchmark);

    // stop the timer
    double time = hTimer.elapsed() * 1000;  // ms

    if (postprocess(&cpu_result, &gold_result, 1))
      printResults(time, iterations);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***Reduction Stage 0***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 0");

    // Calculate Threads per block and total blocks required
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks  = divup(n_elements, dims.dimThreads);

    // Copy input data to device
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Calculate bytes needed for output
    size_t block_bytes = dims.dimBlocks * sizeof(float);

    // Allocate memory for output on device
    float *d_odata = NULL;
    CUDA(cudaMalloc((void **)&d_odata, block_bytes));
    CUDA(cudaMemset(d_odata, 0, block_bytes));

    // Call the kernel. Allocate dynamic shared memory
    reduce_stage0<<<dims.dimBlocks, dims.dimThreads,
                    sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                       n_elements);

    // Copy result of block reduce to CPU and run CPU reduce
    float *h_blocks = (float *)malloc(dims.dimBlocks * sizeof(float));
    CUDA(cudaMemcpy(h_blocks, d_odata, dims.dimBlocks * sizeof(float),
                    cudaMemcpyDeviceToHost));

    // Secondary reduce on CPU
    float gpu_result = 0;
    for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];

    // Check the result and then run the benchmark.
    if (postprocess(&gpu_result, &gold_result, 1)) {
      nvtxRangeId_t rangeBenchmark =
          nvtxRangeStart("Reduction Stage 0 Benchmark");

      // Start Benchmark
      int iterations = 10;
      CUDA(cudaEventRecord(start, 0));

      // Run multiple times for a good benchmark
      for (int i = 0; i < iterations; i++) {
        reduce_stage0<<<dims.dimBlocks, dims.dimThreads,
                        sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                           n_elements);

        cudaMemcpy(h_blocks, d_odata, dims.dimBlocks * sizeof(float),
                   cudaMemcpyDeviceToHost);

        for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];
      }

      CUDA(cudaEventRecord(stop, 0));
      CUDA(cudaEventSynchronize(stop));

      nvtxRangeEnd(rangeBenchmark);

      float time_ms;
      CUDA(cudaEventElapsedTime(&time_ms, start, stop));

      printResults(time_ms, iterations);
    }

    // Cleanup
    free(h_blocks);
    cudaFree(d_odata);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***Reduction Stage 1***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 1");

    // Calculate Threads per block and total blocks required
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks  = divup(n_elements, dims.dimThreads);

    // Copy input data to device
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Calculate bytes needed for output
    size_t block_bytes = dims.dimBlocks * sizeof(float);

    // Allocate memory for output on device
    float *d_odata = NULL;
    CUDA(cudaMalloc((void **)&d_odata, block_bytes));
    CUDA(cudaMemset(d_odata, 0, block_bytes));

    // Call the kernel. Allocate dynamic shared memory
    reduce_stage1<<<dims.dimBlocks, dims.dimThreads,
                    sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                       n_elements);

    // Copy result of block reduce to CPU and run CPU reduce
    float *h_blocks = (float *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

    // Secondary reduce on CPU
    float gpu_result = 0;
    for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];

    // Check the result and then run the benchmark.
    if (postprocess(&gpu_result, &gold_result, 1)) {
      nvtxRangeId_t rangeBenchmark =
          nvtxRangeStart("Reduction Stage 1 Benchmark");

      // Start Benchmark
      int iterations = 10;
      CUDA(cudaEventRecord(start, 0));

      // Run multiple times for a good benchmark
      for (int i = 0; i < iterations; i++) {
        reduce_stage1<<<dims.dimBlocks, dims.dimThreads,
                        sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                           n_elements);

        cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];
      }

      CUDA(cudaEventRecord(stop, 0));
      CUDA(cudaEventSynchronize(stop));

      nvtxRangeEnd(rangeBenchmark);

      float time_ms;
      CUDA(cudaEventElapsedTime(&time_ms, start, stop));

      printResults(time_ms, iterations);
    }

    // Cleanup
    free(h_blocks);
    cudaFree(d_odata);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***Reduction Stage 2***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 2");

    // Calculate Threads per block and total blocks required
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks  = divup(n_elements, dims.dimThreads);

    // Copy input data to device
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Calculate bytes needed for output
    size_t block_bytes = dims.dimBlocks * sizeof(float);

    // Allocate memory for output on device
    float *d_odata = NULL;
    CUDA(cudaMalloc((void **)&d_odata, block_bytes));
    CUDA(cudaMemset(d_odata, 0, block_bytes));

    // Call the kernel. Allocate dynamic shared memory
    reduce_stage2<<<dims.dimBlocks, dims.dimThreads,
                    sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                       n_elements);

    // Copy result of block reduce to CPU and run CPU reduce
    float *h_blocks = (float *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

    // Secondary reduce on CPU
    float gpu_result = 0;
    for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];

    // Check the result and then run the benchmark.
    if (postprocess(&gpu_result, &gold_result, 1)) {
      nvtxRangeId_t rangeBenchmark =
          nvtxRangeStart("Reduction Stage 2 Benchmark");

      // Start Benchmark
      int iterations = 10;
      CUDA(cudaEventRecord(start, 0));

      // Run multiple times for a good benchmark
      for (int i = 0; i < iterations; i++) {
        reduce_stage2<<<dims.dimBlocks, dims.dimThreads,
                        sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                           n_elements);

        cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];
      }

      CUDA(cudaEventRecord(stop, 0));
      CUDA(cudaEventSynchronize(stop));

      nvtxRangeEnd(rangeBenchmark);

      float time_ms;
      CUDA(cudaEventElapsedTime(&time_ms, start, stop));

      printResults(time_ms, iterations);
    }

    // Cleanup
    free(h_blocks);
    cudaFree(d_odata);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***Reduction Stage 3***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 3");

    // Calculate Threads per block and total blocks required
    // Use stage3_TILE in your grid calculation
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks  = divup(n_elements, dims.dimThreads * stage3_TILE);

    // Copy input data to device
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Calculate bytes needed for output
    size_t block_bytes = dims.dimBlocks * sizeof(float);

    // Allocate memory for output on device
    float *d_odata = NULL;
    CUDA(cudaMalloc((void **)&d_odata, block_bytes));
    CUDA(cudaMemset(d_odata, 0, block_bytes));

    // Call the kernel. Allocate dynamic shared memory
    reduce_stage3<<<dims.dimBlocks, dims.dimThreads,
                    sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                       n_elements);

    // Copy result of block reduce to CPU and run CPU reduce
    float *h_blocks = (float *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

    // Secondary reduce on CPU
    float gpu_result = 0;
    for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];

    // Check the result and then run the benchmark.
    if (postprocess(&gpu_result, &gold_result, 1)) {
      nvtxRangeId_t rangeBenchmark =
          nvtxRangeStart("Reduction Stage 3 Benchmark");

      // Start Benchmark
      int iterations = 10;
      CUDA(cudaEventRecord(start, 0));

      // Run multiple times for a good benchmark
      for (int i = 0; i < iterations; i++) {
        reduce_stage3<<<dims.dimBlocks, dims.dimThreads,
                        sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                           n_elements);

        cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];
      }

      CUDA(cudaEventRecord(stop, 0));
      CUDA(cudaEventSynchronize(stop));

      nvtxRangeEnd(rangeBenchmark);

      float time_ms;
      CUDA(cudaEventElapsedTime(&time_ms, start, stop));

      printResults(time_ms, iterations);
    }

    // Cleanup
    free(h_blocks);
    cudaFree(d_odata);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***Reduction Stage 4***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 4");

    // Calculate Threads per block and total blocks required
    // Use stage4_TILE in your grid calculation
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks  = divup(n_elements, dims.dimThreads * stage4_TILE);

    // Copy input data to device
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Calculate bytes needed for output
    size_t block_bytes = dims.dimBlocks * sizeof(float);

    // Allocate memory for output on device
    float *d_odata = NULL;
    CUDA(cudaMalloc((void **)&d_odata, block_bytes));
    CUDA(cudaMemset(d_odata, 0, block_bytes));

    // Call the kernel. Allocate dynamic shared memory
    reduce_stage4<<<dims.dimBlocks, dims.dimThreads,
                    sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                       n_elements);

    // Copy result of block reduce to CPU and run CPU reduce
    float *h_blocks = (float *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

    // Secondary reduce on CPU
    float gpu_result = 0;
    for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];

    // Check the result and then run the benchmark.
    if (postprocess(&gpu_result, &gold_result, 1)) {
      nvtxRangeId_t rangeBenchmark =
          nvtxRangeStart("Reduction Stage 4 Benchmark");

      // Start Benchmark
      int iterations = 10;
      CUDA(cudaEventRecord(start, 0));

      // Run multiple times for a good benchmark
      for (int i = 0; i < iterations; i++) {
        reduce_stage4<<<dims.dimBlocks, dims.dimThreads,
                        sizeof(float) * dims.dimThreads>>>(d_idata, d_odata,
                                                           n_elements);

        cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];
      }

      CUDA(cudaEventRecord(stop, 0));
      CUDA(cudaEventSynchronize(stop));

      nvtxRangeEnd(rangeBenchmark);

      float time_ms;
      CUDA(cudaEventElapsedTime(&time_ms, start, stop));

      printResults(time_ms, iterations);
    }

    // Cleanup
    free(h_blocks);
    cudaFree(d_odata);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "******************************************" << std::endl;
  std::cout << "***Reduction Stage 5***" << std::endl;
  {
    nvtxRangeId_t range = nvtxRangeStart("Reduction Stage 5");

    // Calculate Threads per block and total blocks required
    // Use stage5_TILE in your grid calculation
    DIMS1D dims;
    dims.dimThreads = threads;
    dims.dimBlocks  = divup(n_elements, dims.dimThreads * stage5_TILE);

    // Copy input data to device
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // Calculate bytes needed for output
    size_t block_bytes = dims.dimBlocks * sizeof(float);

    // Allocate memory for output on device
    float *d_odata = NULL;
    CUDA(cudaMalloc((void **)&d_odata, block_bytes));
    CUDA(cudaMemset(d_odata, 0, block_bytes));

    // Call the kernel. Allocate dynamic shared memory
    // Don't forget to add the template
    reduce_stage5<threads>
        <<<dims.dimBlocks, dims.dimThreads, sizeof(float) * dims.dimThreads>>>(
            d_idata, d_odata, n_elements);

    // Copy result of block reduce to CPU and run CPU reduce
    float *h_blocks = (float *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));

    // Secondary reduce on CPU
    float gpu_result = 0;
    for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];

    // Check the result and then run the benchmark.
    if (postprocess(&gpu_result, &gold_result, 1)) {
      nvtxRangeId_t rangeBenchmark =
          nvtxRangeStart("Reduction Stage 5 Benchmark");

      // Start Benchmark
      int iterations = 10;
      CUDA(cudaEventRecord(start, 0));

      // Run multiple times for a good benchmark
      for (int i = 0; i < iterations; i++) {
        reduce_stage5<threads>
            <<<dims.dimBlocks, dims.dimThreads,
               sizeof(float) * dims.dimThreads>>>(d_idata, d_odata, n_elements);

        cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost);

        for (int i = 0; i < dims.dimBlocks; i++) gpu_result += h_blocks[i];
      }

      CUDA(cudaEventRecord(stop, 0));
      CUDA(cudaEventSynchronize(stop));

      nvtxRangeEnd(rangeBenchmark);

      float time_ms;
      CUDA(cudaEventElapsedTime(&time_ms, start, stop));

      printResults(time_ms, iterations);
    }

    // Cleanup
    free(h_blocks);
    cudaFree(d_odata);

    nvtxRangeEnd(range);
  }
  std::cout << "******************************************" << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  // Cleanup
  CUDA(cudaEventDestroy(start));
  CUDA(cudaEventDestroy(stop));

  CUDA(cudaFreeHost(h_idata));
  CUDA(cudaFree(d_idata));

  return 0;
}
