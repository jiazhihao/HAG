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
#ifndef _GNN_H_
#define _GNN_H_
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <cudnn.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <string.h>
#include <map>
#include <set>
#include <vector>

//=====================================================================
// CUDA Helper Functions
//=====================================================================
#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCUDA(status) do {                                         \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int BLOCK_SIZE_LIMIT = 32768;

// CUDA: number of blocks for threads
inline int GET_BLOCKS(const int N)
{
  int ret = (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  return (ret > BLOCK_SIZE_LIMIT) ? BLOCK_SIZE_LIMIT : ret;
}

//====================================================================
// GNN Header Definitions
//====================================================================

typedef int V_ID;
typedef int E_ID;
#define HIDDEN_SIZE 64
#define NUM_LAYERS 2
#define NUM_CLASS 4

struct NodeStruct {
  E_ID index;
};

struct EdgeStruct {
  V_ID src, dst;
};

enum ActMode {
  ACT_MODE_RELU,
};

enum AggMode {
  AGG_MODE_MEAN_POOLING,
};

struct Handler {
  Handler(void);
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  curandGenerator_t gen;
};

struct Graph {
  Graph(void): nv(0), ne(0), rowPtr(NULL), colIdx(NULL) {};
  V_ID nv;
  E_ID ne;
  NodeStruct *rowPtr;
  EdgeStruct *colIdx;
};

class GNNModel {
public:
  GNNModel(Handler handle);
  void set_graph(Graph& graph, int nv,
                 std::map<V_ID, std::set<V_ID>* >& inEdges);
  void set_in_graph(int nv, std::map<V_ID, std::set<V_ID>* >& edgeList);
  void set_out_graph(int nv, std::map<V_ID, std::set<V_ID>* >& edgeList);
  void set_hyper_in_graph(int nv, std::map<V_ID, std::set<V_ID>* >& edgeList);
  void set_hyper_out_graph(int nv, std::map<V_ID, std::set<V_ID>* >& edgeList);

public:
  Handler handle;
  Graph inGraph, outGraph, hyInGraph, hyOutGraph;
};

class Layer {
public:
  Layer(GNNModel* model, Handler handle,
        float* inputPtr, float *inputGradPtr);
  virtual void forward(void) = 0;
  virtual void backward(void) = 0;
  float *outputPtr, *outputGradPtr;
protected:
  GNNModel* model;
  Handler handle;
  float *inputPtr, *inputGradPtr;
};

// hiddenPtr [graph->nv x hiddenDim]
// aggrePtr [graph->nv x hiddenDim]
// outputPtr [graph->nv x outputDim]
class GNNLayer : public Layer {
public:
  GNNLayer(GNNModel* model, Handler handle,
           float* inputPtr, float* inputGradPtr,
           int inputDim, int hiddenDim, int outputDim,
           ActMode act, AggMode agg);
  void forward(void);
  void backward(void);
private:
  int inputDim, hiddenDim, outputDim;
  float *denseWPtr, *denseWGradPtr;
  float *neighWPtr, *neighWGradPtr;
  float *selfWPtr, *selfWGradPtr;
  ActMode actMode;
  AggMode aggMode;
  cudnnActivationDescriptor_t actiDesc;
  cudnnTensorDescriptor_t hiddenTensor, outputTensor;
  float *hiddenPtr, *hiddenGradPtr;
  float *aggrePtr, *aggreGradPtr;
};


// aggrePtr [graph->ng x inputDim]
// outputPtr [graph->ng x numClass]
class GCLayer : public Layer {
public:
  GCLayer(GNNModel* graph, Handler handle,
          float *inputPtr, float* inputGradPtr,
          int inputDim, int numClass);
  void forward(void);
  void backward(void);
private:
  int inputDim, numClass;
  float *denseWPtr, *denseWGradPtr;
  cudnnTensorDescriptor_t outputTensor;
  float *aggrePtr, *aggreGradPtr;
};
#endif
