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
#include "gnn.h"
#include "gnn_kernel.h"

int main()
{
  FILE* file = fopen("BZR_MD/BZR_MD_A.txt", "r");
  V_ID u, v;
  V_ID nv = 0;
  E_ID ne = 0;
  std::map<V_ID, std::set<V_ID>* > inEdges;

  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    if (std::max(u, v) >= nv)
      nv = std::max(u, v) + 1;
    if (inEdges.find(v) == inEdges.end())
      inEdges[v] = new std::set<V_ID>();
    else
      inEdges[v]->insert(u);
  }
  fclose(file);

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
  float* hidden[4];
  for (int i = 0; i < 4; i++)
    checkCUDA(cudaMalloc(&hidden[i], nv * HIDDEN_SIZE * sizeof(float)));
  Graph graph(nv, ne, rowPtrFB, colIdxFB);
  Handler handle;
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
  GNNLayer* layer[NUM_LAYERS];
  for (int i = 0; i < NUM_LAYERS; i++)
    layer[i] = new GNNLayer(&graph, handle, HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE,
                            ACT_MODE_RELU, AGG_MODE_MEAN_POOLING);

  init_weights(hidden[0], nv * HIDDEN_SIZE, 0.05);
  for (int i = 0; i < NUM_LAYERS; i++) {
    layer[i]->forward(hidden[0], hidden[1], hidden[2], hidden[3]);
  }
}

Graph::Graph(int _nv, int _ne, NodeStruct* _rowPtr, EdgeStruct* _colIdx)
: nv(_nv), ne(_ne), rowPtr(_rowPtr), colIdx(_colIdx)
{}

