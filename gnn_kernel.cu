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
void reluBackward(float *gradPtr, const float *input, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    gradPtr[i] = (input[i] > 0.0f) ? gradPtr[i] : 0;
  }
}

__global__
void block_coop_kernel(V_ID rowLeft,
                       V_ID rowRight,
                       int hiddenDim,
                       const NodeStruct* row_ptrs,
                       const EdgeStruct* col_idxs,
                       const float* old_h,
                       float* new_h)
{
  assert(blockDim.x % hiddenDim == 0);
  assert(MIN_HIDDEN_DIM <= hiddenDim);
  int vtxPerBlock = blockDim.x / hiddenDim;
  typedef cub::BlockScan<E_ID, CUDA_NUM_THREADS> BlockScan;
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

GNNLayer::GNNLayer(GNNModel* _model, Handler _handle,
                   float* _inputPtr, float* _inputGradPtr,
                   int _inputDim, int _hiddenDim,
                   int _outputDim,
                   ActMode _actMode, AggMode _aggMode)
: Layer(_model, _handle, _inputPtr, _inputGradPtr),
  inputDim(_inputDim), hiddenDim(_hiddenDim), outputDim(_outputDim),
  actMode(_actMode), aggMode(_aggMode)
{
  int nvSrc = model->inGraph.nvSrc;
  int nvDst = model->inGraph.nvDst;
  // Create and init weights
  // denseW [_inputDim x _hiddenDim]
  checkCUDA(cudaMalloc(&denseWPtr, inputDim * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&denseWGradPtr, inputDim * hiddenDim * sizeof(float)));
  float scale = sqrt(6.0 / (inputDim + hiddenDim));
  init_weights(denseWPtr, inputDim * hiddenDim, scale, handle.gen);
  // neighW [_hiddenDIm x _outputDim]
  checkCUDA(cudaMalloc(&neighWPtr, hiddenDim * outputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&neighWGradPtr, hiddenDim * outputDim * sizeof(float)));
  scale = sqrt(6.0 / (hiddenDim + outputDim));
  init_weights(neighWPtr, hiddenDim * outputDim, scale, handle.gen);
  // selfW [_inputDim x _outputDim]
  checkCUDA(cudaMalloc(&selfWPtr, inputDim * outputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&selfWGradPtr, inputDim * outputDim * sizeof(float)));
  scale = sqrt(6.0 / (inputDim + outputDim));
  init_weights(selfWPtr, inputDim * outputDim, scale, handle.gen);
  // initialize tensors
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnSetActivationDescriptor(actiDesc, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0));
  checkCUDNN(cudnnCreateTensorDescriptor(&hiddenTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(hiddenTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        nvSrc, hiddenDim, 1, 1));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        nvDst, outputDim, 1, 1));
  // allocate hiddenPtr, aggrePtr, outputPtr
  checkCUDA(cudaMalloc(&hiddenPtr, nvSrc * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&hiddenGradPtr, nvSrc * outputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&aggrePtr, nvDst * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&aggreGradPtr, nvDst * hiddenDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputPtr, nvDst * outputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputGradPtr, nvDst * outputDim * sizeof(float)));
}

void print(const float* ptr, int num)
{
  float* ptrZC = (float*) malloc(num * sizeof(float));
  checkCUDA(cudaMemcpy(ptrZC, ptr, num * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < num; i++)
    printf("%.4lf ", ptrZC[i]);
  printf("\n");
  free(ptrZC);
}

void GNNLayer::forward(void)
{
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->inGraph;
  assert(graph.nvSrc == graph.nvDst);
  V_ID nv = graph.nvSrc;
  // Compute hiddenPtr
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        hiddenDim, nv, inputDim,
                        &alpha, denseWPtr, inputDim,
                        inputPtr, inputDim,
                        &beta, hiddenPtr, hiddenDim));
  // relu over hiddenPtr
  checkCUDNN(cudnnActivationForward(handle.dnn, actiDesc, &alpha, hiddenTensor, hiddenPtr, &beta, hiddenTensor, hiddenPtr));
  // Compute aggrePtr
  int blkSize = CUDA_NUM_THREADS / hiddenDim * hiddenDim;
  int numBlks = (nv * hiddenDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  block_coop_kernel<<<numBlks, blkSize>>>(0, nv - 1, hiddenDim,
    graph.rowPtr, graph.colIdx, hiddenPtr, aggrePtr);
  // TODO: add a mean operator in between

  // Compute outputPtr
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        outputDim, nv, hiddenDim,
                        &alpha, neighWPtr, hiddenDim,
                        aggrePtr, hiddenDim,
                        &beta, outputPtr, outputDim));
  // We should add activations on top of the previous Sgemm
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        outputDim, nv, inputDim,
                        &alpha, selfWPtr, inputDim,
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

void GNNLayer::backward(void)
{
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->outGraph;
  assert(graph.nvSrc == graph.nvDst);
  V_ID nv = graph.nvSrc;
  if (actMode == ACT_MODE_RELU) {
    int n = nv * outputDim;
    reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
        outputGradPtr, outputPtr, n);
  } else {
    assert(false);
  }
  // Compute selfWGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        inputDim, outputDim, nv,
                        &alpha, inputPtr, inputDim,
                        outputGradPtr, outputDim,
                        &beta, selfWGradPtr, inputDim));
  // Compute inputGradPtr
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        inputDim, nv, outputDim,
                        &alpha, selfWPtr, inputDim,
                        outputGradPtr, outputDim,
                        &beta, inputGradPtr, inputDim));
  // Compute neighWGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        hiddenDim, outputDim, nv,
                        &alpha, aggrePtr, hiddenDim,
                        outputGradPtr, outputDim,
                        &beta, neighWGradPtr, hiddenDim));
  // Compute aggreGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        hiddenDim, nv, outputDim,
                        &alpha, neighWPtr, hiddenDim,
                        outputGradPtr, outputDim,
                        &beta, aggreGradPtr, hiddenDim));
  // TODO: normalize aggreGradPtr

  // Compute hiddenGrad
  int blkSize = CUDA_NUM_THREADS / hiddenDim * hiddenDim;
  int numBlks = (nv * hiddenDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  block_coop_kernel<<<numBlks, blkSize>>>(0, nv - 1, hiddenDim,
      graph.rowPtr, graph.colIdx, aggreGradPtr, hiddenGradPtr);

  // Backprop relu
  int n = nv * hiddenDim;
  reluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS>>>(
      hiddenGradPtr, hiddenPtr, n);

  // Compute denseWGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        inputDim, hiddenDim, nv,
                        &alpha, inputPtr, inputDim,
                        hiddenGradPtr, hiddenDim,
                        &beta, denseWGradPtr, inputDim));
  // Copute inputGrad
  // Note: this is the second time we compute inputGrad,
  // so we replace beta with alpha
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        inputDim, nv, hiddenDim,
                        &alpha, denseWPtr, inputDim,
                        hiddenGradPtr, hiddenDim,
                        &alpha/**1.0**/,  inputGradPtr, inputDim));
  
}

GCLayer::GCLayer(GNNModel* _model, Handler _handle,
                 float* _inputPtr, float* _inputGradPtr,
                 int _inputDim, int _numClass)
: Layer(_model, _handle, _inputPtr, _inputGradPtr),
  inputDim(_inputDim), numClass(_numClass)
{
  int ng = model->hyInGraph.nvDst;
  // Create and init weights
  // denseW [_inputDim x _numClass]
  checkCUDA(cudaMalloc(&denseWPtr, inputDim * numClass * sizeof(float)));
  checkCUDA(cudaMalloc(&denseWGradPtr, inputDim * numClass * sizeof(float)));
  float scale = sqrt(6.0 / (inputDim + numClass));
  init_weights(denseWPtr, inputDim * numClass, scale, handle.gen);
  // Create aggregate and output tensors
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnSetTensor4dDescriptor(outputTensor, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        ng, numClass, 1, 1));
  // Allocate aggregate and output tensors
  checkCUDA(cudaMalloc(&aggrePtr, ng * inputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputPtr, ng * numClass * sizeof(float)));
  checkCUDA(cudaMalloc(&aggreGradPtr, ng * inputDim * sizeof(float)));
  checkCUDA(cudaMalloc(&outputGradPtr, ng * numClass * sizeof(float)));
}

void GCLayer::forward(void)
{
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->hyInGraph;
  // Compute aggrePtr
  int blkSize = CUDA_NUM_THREADS / inputDim * inputDim;
  int numBlks = (graph.nvDst * inputDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  block_coop_kernel<<<numBlks, blkSize>>>(0, graph.nvDst-1, inputDim,
      graph.rowPtr, graph.colIdx, inputPtr, aggrePtr);
  
  // TODO: normalize graph vector by degrees

  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        numClass, graph.nvDst, inputDim,
                        &alpha, denseWPtr, inputDim,
                        aggrePtr, inputDim,
                        &beta, outputPtr, numClass)); 
  checkCUDNN(cudnnSoftmaxForward(handle.dnn, CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha, outputTensor, aggrePtr,
                                 &beta, outputTensor, outputPtr));
}

void GCLayer::backward(void)
{
  float alpha = 1.0f, beta = 0.0f;
  Graph graph = model->hyOutGraph;
  // Compute denseW grad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        inputDim, numClass, graph.nvDst,
                        &alpha, aggrePtr, inputDim,
                        outputGradPtr, numClass,
                        &beta, denseWGradPtr, inputDim));
  // Compute aggreGrad
  checkCUDA(cublasSgemm(handle.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        inputDim, graph.nvDst, numClass,
                        &alpha, denseWPtr, inputDim,
                        outputGradPtr, numClass,
                        &beta, aggreGradPtr, inputDim));
  // TODO: normalize graph vector by degrees

  int blkSize = CUDA_NUM_THREADS / inputDim * inputDim;
  int numBlks = (graph.nvSrc * inputDim + blkSize - 1) / blkSize;
  if (numBlks > BLOCK_SIZE_LIMIT)
    numBlks = BLOCK_SIZE_LIMIT;
  block_coop_kernel<<<numBlks, blkSize>>>(0, graph.nvSrc, inputDim,
      graph.rowPtr, graph.colIdx, aggreGradPtr, inputGradPtr);
}

Handler::Handler(void)
{
  checkCUDA(cublasCreate(&blas));
  checkCUDNN(cudnnCreate(&dnn));
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

__global__
void scale_kernel(float* ptr, int size, float a, float b)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__
void seq_kernel(float* ptr, int size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = 1;
  }
}

void init_weights(float* ptr, int num, float scale, curandGenerator_t genGPU)
{
  curandGenerateUniform(genGPU, ptr, num);
  scale_kernel<<<GET_BLOCKS(num), CUDA_NUM_THREADS>>>(
      ptr, num, -scale, scale);
}

void seq_weights(float* ptr, int num)
{
  seq_kernel<<<GET_BLOCKS(num), CUDA_NUM_THREADS>>>(ptr, num);
}
