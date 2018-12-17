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
  std::map<V_ID, std::set<V_ID>* > inEdges, outEdges;

  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    ne ++;
    if (std::max(u, v) >= nv)
      nv = std::max(u, v) + 1;
    // add inEdge
    if (inEdges.find(v) == inEdges.end())
      inEdges[v] = new std::set<V_ID>();
    inEdges[v]->insert(u);
    // add outEdge
    if (outEdges.find(u) == outEdges.end())
      outEdges[u] = new std::set<V_ID>();
    outEdges[u]->insert(v);
  }
  fclose(file);
  printf("nv(%d) ne (%d)\n", nv, ne);
  GNNModel model(handle);
  model.set_in_graph(nv, inEdges);
  model.set_out_graph(nv, outEdges);

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

  std::vector<Layer*> layers;
  for (int i = 0; i < NUM_LAYERS; i++) {
    float* inputPtr = (i == 0) ? hidden[0] : layers[i-1]->outputPtr;
    float* inputGradPtr = (i == 0) ? NULL : layers[i-1]->outputGradPtr;
    layers.push_back(new GNNLayer(&model, handle, inputPtr, inputGradPtr,
                             HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE,
                             ACT_MODE_RELU, AGG_MODE_MEAN_POOLING));
  }
  GCLayer* gcLayer = new GCLayer(&model, handle,
                                 layers[layers.size()-1]->outputPtr,
                                 layers[layers.size()-1]->outputGradPtr,
                                 HIDDEN_SIZE, NUM_CLASS);

  for (int i = 0; i < layers.size(); i++) {
    layers[i]->forward();
  }
}

GNNModel::GNNModel(Handler _handle)
: handle(_handle) {}

void GNNModel::set_graph(Graph& graph, int nv,
                         std::map<V_ID, std::set<V_ID>* >& inEdges)
{
  graph.nv = nv;
  graph.ne = 0;
  for (int v = 0; v < nv; v++)
    if (inEdges.find(v) != inEdges.end())
      graph.ne += inEdges[v]->size();
  NodeStruct *rowPtrZC, *rowPtrFB;
  EdgeStruct *colIdxZC, *colIdxFB;
  rowPtrZC = (NodeStruct*) malloc(graph.nv * sizeof(NodeStruct));
  colIdxZC = (EdgeStruct*) malloc(graph.ne * sizeof(EdgeStruct));
  E_ID count = 0;
  for (int v = 0; v < nv; v++) {
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
  checkCUDA(cudaMalloc(&rowPtrFB, graph.nv * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&colIdxFB, graph.ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMemcpy(rowPtrFB, rowPtrZC, graph.nv * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(colIdxFB, colIdxZC, graph.ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  free(rowPtrZC);
  free(colIdxZC);
  graph.rowPtr = rowPtrFB;
  graph.colIdx = colIdxFB;
}

void GNNModel::set_in_graph(int nv,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(inGraph, nv, edgeList);
  printf("Add normal in-edge graph: nv(%d) ne(%d)\n", inGraph.nv, inGraph.ne);
}

void GNNModel::set_out_graph(int nv,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(outGraph, nv, edgeList);
  printf("Add normal out-edge graph: nv(%d) ne(%d)\n", outGraph.nv, outGraph.ne);
}

void GNNModel::set_hyper_in_graph(int nv,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(hyInGraph, nv, edgeList);
  printf("Add hyper in-edge graph: nv(%d) ne(%d)\n", hyInGraph.nv, hyInGraph.ne);
}

void GNNModel::set_hyper_out_graph(int nv,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(hyOutGraph, nv, edgeList);
  printf("Add hyper in-edge graph: nv(%d) ne(%d)\n", hyOutGraph.nv, hyOutGraph.ne);
}

Layer::Layer(GNNModel* _model, Handler _handle,
             float* _inputPtr, float* _inputGradPtr)
: model(_model), handle(_handle),
  inputPtr(_inputPtr), inputGradPtr(_inputGradPtr)
{}
