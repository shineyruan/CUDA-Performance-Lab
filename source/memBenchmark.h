#ifndef CUDABENCHMARK_H
#define CUDABENCHMARK_H

#include <assert.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <stdio.h>

#include <chrono>
#include <cstring>
#include <thread>

static const unsigned int MB_TO_TRANSFER = 16;

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

class Timer {
public:
  Timer() : beg_(clock_::now()) {}
  void reset() { beg_ = clock_::now(); }
  double elapsed() const {
    return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};

void profileH2HCopies(float *h_a, float *h_b, unsigned int n,
                      const char *desc) {
  printf("\nHost to Host %s Memcpy\n", desc);

  unsigned int bytes = n * sizeof(float);

  // Warm Up
  for (int i = 0; i < 32; i++) memcpy(h_a, h_b, bytes);

  int iters = 100;
  Timer timer;
  for (int i = 0; i < iters; i++) memcpy(h_a, h_b, bytes);

  float time1 = timer.elapsed();
  float band1 =
      2.0f * iters * bytes * (float)1e-9 / time1;  // 2.0 for read and write

  timer.reset();
  for (int i = 0; i < iters; i++) memcpy(h_b, h_a, bytes);

  float time2 = timer.elapsed();
  float band2 = 2.0f * iters * bytes * (float)1e-9 / time2;

  printf("  Host to Host bandwidth (GB/s): %.2f\n", (band1 + band2) / 2.0f);
}

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n,
                   const char *desc) {
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  // Warm Up
  for (int i = 0; i < 16; i++) {
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
  }

  // events for timing
  cudaEvent_t startEvent, stopEvent;

  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  float time;
  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("  Host to Device bandwidth (GB/s): %.2f\n", bytes * 1e-6 / time);

  checkCuda(cudaEventRecord(startEvent, 0));
  checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
  printf("  Device to Host bandwidth (GB/s): %.2f\n", bytes * 1e-6 / time);

  for (unsigned int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***", desc);
      break;
    }
  }

  // clean up events
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
}

void profileD2DCopies(float *d_a, float *d_b, unsigned int n) {
  printf("\nDevice to Device Memcpy\n");

  unsigned int bytes = n * sizeof(float);

  // Warm Up
  for (int i = 0; i < 16; i++) {
    checkCuda(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));
    checkCuda(cudaMemcpy(d_a, d_b, bytes, cudaMemcpyDeviceToDevice));
  }

  // events for timing
  cudaEvent_t startEvent, stopEvent;
  int iters = 100;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < iters; i++)
    checkCuda(cudaMemcpy(d_b, d_a, bytes, cudaMemcpyDeviceToDevice));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  float time1;
  checkCuda(cudaEventElapsedTime(&time1, startEvent, stopEvent));
  float band1 =
      2.0f * iters * bytes * (float)1e-6 / time1;  // 2.0 for read and write

  checkCuda(cudaEventRecord(startEvent, 0));
  for (int i = 0; i < iters; i++)
    checkCuda(cudaMemcpy(d_a, d_b, bytes, cudaMemcpyDeviceToDevice));
  checkCuda(cudaEventRecord(stopEvent, 0));
  checkCuda(cudaEventSynchronize(stopEvent));

  float time2;
  checkCuda(cudaEventElapsedTime(&time2, startEvent, stopEvent));

  float band2 = 2.0f * iters * bytes * (float)1e-6 / time2;
  printf("  Device to Device bandwidth (GB/s): %.2f\n", (band1 + band2) / 2.0f);

  float *h_a;
  float *h_b;

  checkCuda(cudaMallocHost((void **)&h_a, bytes));  // host pinned
  checkCuda(cudaMallocHost((void **)&h_b, bytes));  // host pinned

  checkCuda(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
  checkCuda(cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost));

  for (unsigned int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** Device to device transfers failed ***");
      break;
    }
  }

  // clean up events
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
}

int memBenchmark() {
  unsigned int nElements   = MB_TO_TRANSFER * 256 * 1024;
  const unsigned int bytes = nElements * sizeof(float);

  // host arrays
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  // device array
  float *d_a;
  float *d_b;

  // allocate and initialize
  h_aPageable = (float *)malloc(bytes);                   // host pageable
  h_bPageable = (float *)malloc(bytes);                   // host pageable
  checkCuda(cudaMallocHost((void **)&h_aPinned, bytes));  // host pinned
  checkCuda(cudaMallocHost((void **)&h_bPinned, bytes));  // host pinned
  checkCuda(cudaMalloc((void **)&d_a, bytes));            // device
  checkCuda(cudaMalloc((void **)&d_b, bytes));            // device

  for (unsigned int i = 0; i < nElements; ++i) h_aPageable[i] = (float)i;

  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;
  checkCuda(cudaGetDeviceProperties(&prop, 0));

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // perform copies and report bandwidth
  nvtxRangeId_t h2hPageable_range =
      nvtxRangeStart("Host to Host Paged Memory Transfer");
  profileH2HCopies(h_aPageable, h_bPageable, nElements, "Pageable");
  nvtxRangeEnd(h2hPageable_range);

  nvtxRangeId_t h2hPinned_range =
      nvtxRangeStart("Host to Host Pinned Memory Transfer");
  profileH2HCopies(h_aPinned, h_bPinned, nElements, "Pinned");
  nvtxRangeEnd(h2hPinned_range);

  nvtxRangeId_t pageable_range = nvtxRangeStart("Paged Memory Transfer");
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  nvtxRangeEnd(pageable_range);

  nvtxRangeId_t pinned_range = nvtxRangeStart("Pinned Memory Transfer");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
  nvtxRangeEnd(pinned_range);

  nvtxRangeId_t d2d_range = nvtxRangeStart("Device to Device Memory Transfer");
  profileD2DCopies(d_a, d_b, nElements);
  nvtxRangeEnd(d2d_range);

  printf("\n");

  // cleanup
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  cudaDeviceReset();

  return 0;
}

#endif
