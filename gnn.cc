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

#include <map>
#include <set>
#include <vector>
#include "gnn.h"
#include "gnn_kernel.h"

int main()
{
  Handler handle;
  //FILE* file = fopen("BZR_MD/BZR_MD_A.txt", "r");
  FILE* file = fopen("input.txt", "r");
  V_ID u, v;
  V_ID nv = 0;
  E_ID ne = 0;
  std::map<V_ID, std::set<V_ID>* > inEdges;

  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    ne ++;
    if (std::max(u, v) >= nv)
      nv = std::max(u, v) + 1;
    if (inEdges.find(v) == inEdges.end())
      inEdges[v] = new std::set<V_ID>();
    inEdges[v]->insert(u);
  }
  fclose(file);
  printf("nv(%d) ne (%d)\n", nv, ne);

  NodeStruct *rowPtrZC, *rowPtrFB;
  EdgeStruct *colIdxZC, *colIdxFB;
  rowPtrZC = (NodeStruct*) malloc(nv * sizeof(NodeStruct));
  colIdxZC = (EdgeStruct*) malloc(ne * sizeof(EdgeStruct));
  E_ID count = 0;
  for (v = 0; v < nv; v++) {
    if (inEdges.find(v) != inEdges.end()) {
      std::set<V_ID>::const_iterator it;
      for (it = inEdges[v]->begin(); it != inEdges[v]->end(); it++) {
        colIdxZC[count].src = *it;
        colIdxZC[count].dst = v;
        count ++;
      }
    }
    rowPtrZC[v].index = count;
  }
  checkCUDA(cudaMalloc(&rowPtrFB, nv * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&colIdxFB, ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMemcpy(rowPtrFB, rowPtrZC, nv * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(colIdxFB, colIdxZC, ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  float* hidden[4];
  for (int i = 0; i < 4; i++)
    checkCUDA(cudaMalloc(&hidden[i], nv * HIDDEN_SIZE * sizeof(float)));
  init_weights(hidden[0], nv * HIDDEN_SIZE, 0.5, handle.gen);
  int ng = nv;
  GNNModel graph(nv, ne, ng, rowPtrFB, colIdxFB, rowPtrFB, colIdxFB, rowPtrFB, colIdxFB, rowPtrFB, colIdxFB);

  std::vector<Layer*> layers;
  for (int i = 0; i < NUM_LAYERS; i++) {
    float* inputPtr = (i == 0) ? hidden[0] : layers[i-1]->outputPtr;
    float* inputGradPtr = (i == 0) ? NULL : layers[i-1]->outputGradPtr;
    layers.push_back(new GNNLayer(&graph, handle, inputPtr, inputGradPtr,
                             HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE,
                             ACT_MODE_RELU, AGG_MODE_MEAN_POOLING));
  }
  GCLayer* gcLayer = new GCLayer(&graph, handle,
                                 layers[layers.size()-1]->outputPtr,
                                 layers[layers.size()-1]->outputGradPtr,
                                 HIDDEN_SIZE, NUM_CLASS);

  for (int i = 0; i < layers.size(); i++) {
    layers[i]->forward();
  }
}

GNNModel::GNNModel(V_ID _nv, E_ID _ne, V_ID _ng,
             NodeStruct* _inRowPtr, EdgeStruct* _inColIdx,
             NodeStruct* _outRowPtr, EdgeStruct* _outColIdx,
             NodeStruct* _grInRowPtr, EdgeStruct* _grInColIdx,
             NodeStruct* _grOutRowPtr, EdgeStruct* _grOutColIdx)
: nv(_nv), ne(_ne), ng(_ng),
  inRowPtr(_inRowPtr), inColIdx(_inColIdx),
  outRowPtr(_outRowPtr), outColIdx(_outColIdx),
  grInRowPtr(_grInRowPtr), grInColIdx(_grInColIdx),
  grOutRowPtr(_grOutRowPtr), grOutColIdx(_grOutColIdx)
{}

Layer::Layer(GNNModel* _graph, Handler _handle,
             float* _inputPtr, float* _inputGradPtr)
: graph(_graph), handle(_handle),
  inputPtr(_inputPtr), inputGradPtr(_inputGradPtr)
{}
