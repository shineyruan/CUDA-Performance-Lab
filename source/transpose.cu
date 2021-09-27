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
#include <string>

// Initialize sizes
const int sizeX = 1234;
const int sizeY = 3153;

struct DIMS {
  dim3 dimBlock;
  dim3 dimGrid;
};

#define CUDA(call)                                       \
  do {                                                   \
    cudaError_t e = (call);                              \
    if (e == cudaSuccess) break;                         \
    fprintf(stderr, __FILE__ ":%d: %s (%d)\n", __LINE__, \
            cudaGetErrorString(e), e);                   \
    exit(1);                                             \
  } while (0)

// This function divides up the n by div - similar to ceil
// Example, divup(10, 3) = 4
inline unsigned divup(unsigned n, unsigned div) { return (n + div - 1) / div; }

void printResults(double timeInMilliseconds, int iterations) {
  // print out the time required for the kernel to finish the transpose
  // operation
  double bandwidth =
      (iterations * 2 * 1000 * (double)(sizeX * sizeY * sizeof(float))) /
      (1000 * 1000 * 1000 * timeInMilliseconds);
  std::cout << "Elapsed Time for " << iterations
            << " runs = " << round(timeInMilliseconds) << "ms" << std::endl;
  std::cout << termcolor::bold << termcolor::red << termcolor::on_white
            << "Bandwidth (GB/s) = " << std::setprecision(4) << bandwidth
            << termcolor::reset << std::endl;
}

// Check errors
bool postprocess(const float* ref, const float* res, int n) {
  bool passed = true;
  for (int i = 0; i < n; i++) {
    if (res[i] != ref[i]) {
      std::cout << "ID: " << i << " \t Res: " << res[i] << " \t Ref: " << ref[i]
                << std::endl;
      std::cout << termcolor::blink << termcolor::white << termcolor::on_red
                << "*** FAILED ***" << termcolor::reset << std::endl;
      passed = false;
      break;
    }
  }

  if (passed) {
    std::cout << termcolor::green << "Post process check passed!!"
              << termcolor::reset << std::endl;
  }

  return passed;
}

void preprocess(float* res, float* dev_res, int n) {
  std::fill(res, res + n, -1);
  cudaMemset(dev_res, -1, n * sizeof(float));
}

// TODO: COMPLETE THIS
__global__ void copyKernel(const float* const a, float* const b) {
  int i = 0;  // Compute correctly - Global X index
  int j = 0;  // Compute correctly - Global Y index

  // Check if i or j are out of bounds. If they are, return.

  int index = 0;  // Compute 1D index from i, j

  // Copy data from A to B
  b[index] = a[index];
}

// TODO: COMPLETE THIS
__global__ void matrixTransposeNaive(const float* const a, float* const b) {
  int i = 0;  // Compute correctly - Global X index
  int j = 0;  // Compute correctly - Global Y index

  // Check if i or j are out of bounds. If they are, return.

  int index_in  = 0;  // Compute input index (i,j) from matrix A
  int index_out = 0;  // Compute output index (j,i) in matrix B = transpose(A)

  // Copy data from A to B using transpose indices
  b[index_out] = a[index_in];
}

// TODO: COMPLETE THIS
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void matrixTransposeShared(const float* const a, float* const b) {
  // Allocate appropriate shared memory - use mat as variable name
  // Example: <shared specifier> type mat[size][size]; - replace size with the
  // correct values

  // Compute input and output index
  int bx = 0;  // Compute block offset - this is number of global threads in X
               // before this block
  int by = 0;  // Compute block offset - this is number of global threads in Y
               // before this block
  int i = 0;   // Global input x index - Same as previous kernels
  int j = 0;   // Global input y index - Same as previous kernels

  // We are transposing the blocks here. See how ti uses by and tj uses bx
  // We transpose blocks using indices, and transpose with block sub-matrix
  // using the shared memory
  int ti = 0;  // Global output x index - remember the transpose
  int tj = 0;  // Global output y index - remember the transpose

  // Copy data from input to shared memory

  // Copy data from shared memory to global memory
  // Switch threadIdx.x and threadIdx.y from input read
}

template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void matrixTransposeSharedwBC(const float* const a, float* const b) {
  // Allocate appropriate shared memory

  // Compute input and output index - same as matrixTransposeShared kernel
  int bx = 0;  // Compute block offset - this is number of global threads in X
               // before this block
  int by = 0;  // Compute block offset - this is number of global threads in Y
               // before this block
  int i = 0;   // Global input x index - Same as previous kernels
  int j = 0;   // Global input y index - Same as previous kernels

  // We are transposing the blocks here. See how ti uses by and tj uses bx
  // We transpose blocks using indices, and transpose with block sub-matrix
  // using the shared memory
  int ti = 0;  // Global output x index - remember the transpose
  int tj = 0;  // Global output y index - remember the transpose

  // Copy data from input to shared memory - similar to matrixTransposeShared
  // Kernel

  // Copy data from shared memory to global memory - similar to
  // matrixTransposeShared Kernel
}

template <int TILE, int SIDE>
__global__ void matrixTransposeUnrolled(const float* a, float* b) {
  // Allocate appropriate shared memory

  // Compute input and output index
  int x = 0;
  int y = 0;

  // Copy data from input to shared memory. Multiple copies per thread.

  // Copy data from shared memory to global memory. Multiple copies per thread.
}

int main(int argc, char* argv[]) {
  // Run Memcpy benchmarks
  nvtxRangeId_t cudaBenchmark = nvtxRangeStart("CUDA Memcpy Benchmark");
  memBenchmark();
  nvtxRangeEnd(cudaBenchmark);

  // Host arrays.
  float* a      = new float[sizeX * sizeY];
  float* b      = new float[sizeX * sizeY];
  float* a_gold = new float[sizeX * sizeY];
  float* b_gold = new float[sizeX * sizeY];

  // Device arrays
  float *d_a, *d_b;

  // Allocate memory on the device
  CUDA(cudaMalloc((void**)&d_a, sizeX * sizeY * sizeof(float)));

  CUDA(cudaMalloc((void**)&d_b, sizeX * sizeY * sizeof(float)));

  // Fill matrix A
  for (int i = 0; i < sizeX * sizeY; i++) a[i] = (float)i;

  // Copy array contents of A from the host (CPU) to the device (GPU)
  cudaMemcpy(d_a, a, sizeX * sizeY * sizeof(float), cudaMemcpyHostToDevice);

  // Compute "gold" reference standard
  for (int jj = 0; jj < sizeY; jj++) {
    for (int ii = 0; ii < sizeX; ii++) {
      a_gold[jj * sizeX + ii] = a[jj * sizeX + ii];
      b_gold[ii * sizeY + jj] = a[jj * sizeX + ii];
    }
  }

  std::cout << std::endl;

  cudaDeviceSynchronize();

  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

#define CPU_TRANSPOSE
#ifdef CPU_TRANSPOSE
  ////////////////////////////////////////////////////////////
  std::cout << "****************************************************"
            << std::endl;
  std::cout << "***CPU Transpose***" << std::endl;
  {
    // start the timer
    nvtxRangeId_t cpuBenchmark = nvtxRangeStart("CPU Transpose Benchmark");

    Timer hTimer;
    int iterations = 10;
    for (int k = 0; k < iterations; k++) {
      for (int jj = 0; jj < sizeY; jj++)
        for (int ii = 0; ii < sizeX; ii++)
          b[ii * sizeX + jj] = a[jj * sizeX + ii];
    }
    double time = hTimer.elapsed() * 1000;  // ms

    nvtxRangeEnd(cpuBenchmark);

    printResults(time, iterations);
  }
  std::cout << "****************************************************"
            << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////
#endif

  ////////////////////////////////////////////////////////////
  std::cout << "****************************************************"
            << std::endl;
  std::cout << "***Device To Device Copy***" << std::endl;
  {
    preprocess(b, d_b, sizeX * sizeY);

    // TODO: COMPLETE THIS
    // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid"
    DIMS dims;
    dims.dimBlock = dim3(1, 1, 1);
    dims.dimGrid  = dim3(1, 1, 1);

    // start the timer
    nvtxRangeId_t copyKernelBenchmark = nvtxRangeStart("Device to Device Copy");
    cudaEventRecord(start, 0);

    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
      // Launch the GPU kernel
      copyKernel<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
    }
    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    nvtxRangeEnd(copyKernelBenchmark);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, d_b, sizeY * sizeX * sizeof(float), cudaMemcpyDeviceToHost);

    if (postprocess(a_gold, b, sizeX * sizeY)) {
      printResults(time, iterations);
    }
  }
  std::cout << "****************************************************"
            << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "****************************************************"
            << std::endl;
  std::cout << "***Naive Transpose***" << std::endl;
  {
    preprocess(b, d_b, sizeX * sizeY);

    // TODO: COMPLETE THIS
    // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid"
    DIMS dims;
    dims.dimBlock = dim3(1, 1, 1);
    dims.dimGrid  = dim3(1, 1, 1);

    nvtxRangeId_t naiveTransposeBenchmark =
        nvtxRangeStart("Naive Transpose Benchmark");
    cudaEventRecord(start, 0);

    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
      // Launch the GPU kernel
      matrixTransposeNaive<<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
    }
    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    nvtxRangeEnd(naiveTransposeBenchmark);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, d_b, sizeY * sizeX * sizeof(float), cudaMemcpyDeviceToHost);

    if (postprocess(b_gold, b, sizeX * sizeY)) {
      printResults(time, iterations);
    }
  }
  std::cout << "****************************************************"
            << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "****************************************************"
            << std::endl;
  std::cout << "***Shared Memory Transpose***" << std::endl;
  {
    preprocess(b, d_b, sizeX * sizeY);

    // TODO: COMPLETE THIS
    // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid"
    DIMS dims;
    dims.dimBlock = dim3(1, 1, 1);
    dims.dimGrid  = dim3(1, 1, 1);

    nvtxRangeId_t sharedMemoryTransposeBenchmark =
        nvtxRangeStart("Shared Memory Transpose Benchmark");
    cudaEventRecord(start, 0);

    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
      // Launch the GPU kernel
      matrixTransposeShared<32, 32><<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
    }
    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    nvtxRangeEnd(sharedMemoryTransposeBenchmark);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, d_b, sizeY * sizeX * sizeof(float), cudaMemcpyDeviceToHost);

    if (postprocess(b_gold, b, sizeX * sizeY)) {
      printResults(time, iterations);
    }
  }
  std::cout << "****************************************************"
            << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "****************************************************"
            << std::endl;
  std::cout << "***Shared Memory Transpose without Bank Conflicts***"
            << std::endl;
  {
    preprocess(b, d_b, sizeX * sizeY);

    // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid"
    DIMS dims;
    dims.dimBlock = dim3(1, 1, 1);
    dims.dimGrid  = dim3(1, 1, 1);

    nvtxRangeId_t sharedMemoryTransposeWBCBenchmark = nvtxRangeStart(
        "Shared Memory Transpose Without Bank Conflict Benchmark");
    cudaEventRecord(start, 0);

    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
      // Launch the GPU kernel
      matrixTransposeSharedwBC<1, 1><<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
    }
    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    nvtxRangeEnd(sharedMemoryTransposeWBCBenchmark);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, d_b, sizeY * sizeX * sizeof(float), cudaMemcpyDeviceToHost);

    if (postprocess(b_gold, b, sizeX * sizeY)) {
      printResults(time, iterations);
    }
  }
  std::cout << "****************************************************"
            << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////
  std::cout << "****************************************************"
            << std::endl;
  std::cout << "***Unrolled Loop Transpose***" << std::endl;
  {
    preprocess(b, d_b, sizeX * sizeY);

    // Assign a 2D distribution of BS_X x BS_Y x 1 CUDA threads within
    // Calculate number of blocks along X and Y in a 2D CUDA "grid"
    const int tile = 32;
    const int side = 8;
    DIMS dims;
    dims.dimBlock = dim3(tile, side, 1);
    dims.dimGrid  = dim3(divup(sizeX, tile), divup(sizeY, tile), 1);

    nvtxRangeId_t sharedMemoryTransposeWBCBenchmark = nvtxRangeStart(
        "Shared Memory Transpose Without Bank Conflict Benchmark");
    cudaEventRecord(start, 0);

    int iterations = 10;
    for (int i = 0; i < iterations; i++) {
      // Launch the GPU kernel
      matrixTransposeUnrolled<tile, side>
          <<<dims.dimGrid, dims.dimBlock>>>(d_a, d_b);
    }
    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    nvtxRangeEnd(sharedMemoryTransposeWBCBenchmark);

    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b, d_b, sizeY * sizeX * sizeof(float), cudaMemcpyDeviceToHost);

    if (postprocess(b_gold, b, sizeX * sizeY)) {
      printResults(time, iterations);
    }
  }
  std::cout << "****************************************************"
            << std::endl
            << std::endl;
  ////////////////////////////////////////////////////////////

  // free device memory
  cudaFree(d_a);
  cudaFree(d_b);

  // free host memory
  delete[] a;
  delete[] b;

  // CUDA Reset for NVProf
  CUDA(cudaDeviceReset());

  // successful program termination
  return 0;
}
