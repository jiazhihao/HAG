/* Copyright 2019 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>
#include "gnn_kernel.h"
#define MIN_HIDDEN_DIM 32

__global__
void block_coop_kernel(V_ID rowLeft,
                       V_ID rowRight,
                       int hiddenDim,
                       const NodeStruct* row_ptrs,
                       const EdgeStruct* col_idxs,
                       float* old_h,
                       float* new_h)
{
  assert(blockDim.x % hiddenDim == 0);
  assert(MIN_HIDDEN_DIM <= hiddenDim);
  int vtxPerBlock = blockDim.x / hiddenDim;
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS / MIN_HIDDEN_DIM> BlockScan;
  __shared__ BlockScan::TempStorage temp_storage;
  __shared__ E_ID blkColStart;
  __shared__ float acc_h[CUDA_NUM_THREADS];
  int tidDiv = threadIdx.x / hiddenDim;
  int tidMod = threadIdx.x % hiddenDim;
  for (V_ID blkRowStart = blockIdx.x * vtxPerBlock + rowLeft;
       blkRowStart <= rowRight;
       blkRowStart += vtxPerBlock * gridDim.x)
  {
    E_ID myNumEdges = 0, scratchOffset, totalNumEdges = 0;
    if (threadIdx.x + blkRowStart <= rowRight && threadIdx.x < vtxPerBlock) {
      V_ID curVtx = threadIdx.x + blkRowStart;
      E_ID startColIdx, endColIdx = row_ptrs[curVtx].index;
      if (curVtx == 0)
        startColIdx = 0;
      else
        startColIdx = row_ptrs[curVtx-1].index;
      myNumEdges = endColIdx - startColIdx;
      if (threadIdx.x == 0)
        blkColStart = startColIdx;
    }
    acc_h[threadIdx.x] = 0.0f;
    __syncthreads();
    BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
    E_ID done = 0;
    while (totalNumEdges > 0) {
      if (tidDiv < totalNumEdges) {
        EdgeStruct es = col_idxs[blkColStart + done + tidDiv];
        float val = old_h[es.src * hiddenDim + tidMod];
        int offset = (es.dst - blkRowStart) * hiddenDim + tidMod;
        atomicAdd(&acc_h[offset], val);
      }
      done += vtxPerBlock;
      totalNumEdges -= (totalNumEdges > vtxPerBlock) ? vtxPerBlock : totalNumEdges;
    }
    __syncthreads();
    if (tidDiv + blkRowStart <= rowRight)
      new_h[blkRowStart + threadIdx.x] = acc_h[threadIdx.x];
  }
}

GNNLayer::GNNLayer(Graph* _graph, Handler _handle,
                   int _inputDim, int _hiddenDim,
                   int _outputDim,
                   ActMode _actMode, AggMode _aggMode)
: graph(_graph), handle(_handle), 
  inputDim(_inputDim), hiddenDim(_hiddenDim), outputDim(_outputDim),
  actMode(_actMode), aggMode(_aggMode)
{
  // Create and init weights
  // denseW [_inputDim x _hiddenDim]
  checkCUDA(cudaMalloc(&denseW, inputDim * hiddenDim * sizeof(float)));
  float scale = sqrt(6.0 / (inputDim + hiddenDim));
  init_weights(denseW, inputDim * hiddenDim, scale);
  // neighW [_hiddenDIm x _outputDim]
  checkCUDA(cudaMalloc(&neighW, hiddenDim * outputDim * sizeof(float)));
  scale = sqrt(6.0 / (hiddenDim + outputDim));
  init_weights(neighW, hiddenDim * outputDim, scale);
  // selfW [_inputDim x _outputDim]
  checkCUDA(cudaMalloc(&selfW, inputDim * outputDim * sizeof(float)));
  scale = sqrt(6.0 / (inputDim + outputDim));
  init_weights(selfW, inputDim * outputDim, scale);
  // initialize tensors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0));
  checkCUDNN(cudnnCreateTensorDescriptor(&hiddenTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(hiddenTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        graph->nv, hiddenDim, 1, 1));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        graph->nv, outputDim, 1, 1));
}

void GNNLayer::forward(const float* inputPtr,
                       float* hiddenPtr,
                       float* aggrePtr,
                       float* outputPtr)
{
  float alpha = 1.0f, beta = 0.0f;
  // Compute hiddenPtr
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        hiddenDim, graph->nv, inputDim,
                        &alpha, denseW, inputDim,
                        inputPtr, inputDim,
                        &beta, hiddenPtr, hiddenDim));
  // relu over hiddenPtr
  checkCUDNN(cudnnActivationForward(handle.dnn, actiDesc, &alpha, hiddenTensor, hiddenPtr, &beta, hiddenTensor, hiddenPtr));
  // Compute aggrePtr
  int blkSize = CUDA_NUM_THREADS / hiddenDim * hiddenDim;
  int numBlks = (graph->nv * hiddenDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  block_coop_kernel<<<numBlks, blkSize>>>(0, graph->nv - 1, hiddenDim,
                                          graph->rowPtr, graph->colIdx, hiddenPtr, aggrePtr);
  checkCUDA(cudaDeviceSynchronize());
  // Compute outputPtr
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        outputDim, graph->nv, hiddenDim,
                        &alpha, neighW, hiddenDim,
                        aggrePtr, hiddenDim,
                        &beta, outputPtr, outputDim));
  // We should add activations on top of the previous Sgemm
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        outputDim, graph->nv, inputDim,
                        &alpha, selfW, inputDim,
                        inputPtr, inputDim,
                        &alpha, outputPtr, outputDim));
  if (actMode == ACT_MODE_RELU) {
    checkCUDNN(cudnnActivationForward(handle.dnn, actiDesc,
                                      &alpha, outputTensor, outputPtr,
                                      &beta, outputTensor, outputPtr));
  } else {
    assert(false);
  }
}

__global__
void scale_kernel(float* ptr, int size, float a, float b)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

void init_weights(float* ptr, int num, float scale)
{
  curandGenerator_t genGPU;
  curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(genGPU, 1234ULL);
  curandGenerateUniform(genGPU, ptr, num);
  scale_kernel<<<GET_BLOCKS(num), CUDA_NUM_THREADS>>>(
      ptr, num, -scale, scale);
  curandDestroyGenerator(genGPU);
}
