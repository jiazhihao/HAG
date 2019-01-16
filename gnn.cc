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

void parse_input_args(char **argv, int argc,
                      std::string &graphFile, std::string &hyGraphFile,
                      std::string &nodeLabelFile, std::string &graphLabelFile,
                      double &learningRate, int &epochs)
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "-g"))
    {
      graphFile = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-hg"))
    {
      hyGraphFile = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-nl"))
    {
      nodeLabelFile = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-lr"))
    {
      learningRate = std::atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "-e"))
    {
      epochs = std::atoi(argv[++i]);
      continue;
    }
  }
}

int main(int argc, char **argv)
{
  std::string graphFile, hyGraphFile, nodeLabelFile, graphLabelFile;
  double learningRate = 0.001f;
  int epochs = 100;
  V_ID maxDepth = 10;
  V_ID maxWidth = 60000;
  parse_input_args(argv, argc, graphFile, hyGraphFile,
                   nodeLabelFile, graphLabelFile,
                   learningRate, epochs);
  Handler handle;
  //FILE* file = fopen("BZR_MD/BZR_MD_A.txt", "r");
  FILE* file = fopen(graphFile.c_str(), "r");
  V_ID u, v;
  V_ID nv = 0;
  std::map<V_ID, std::set<V_ID>* > inEdges;
  E_ID ne = 0;
  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    // shift node indices by 1 to make them 0-indexed
    u --;
    // shift node indices by 1 to make them 0-indexed
    v --;
    ne ++;
    if (std::max(u, v) >= nv)
      nv = std::max(u, v) + 1;
    // add inEdge
    if (inEdges.find(v) == inEdges.end())
      inEdges[v] = new std::set<V_ID>();
    inEdges[v]->insert(u);
  }
  //int cnt = 0;
  //for (v = 0; v < nv; v++)
  //  if (inEdges.find(v) != inEdges.end()) {
  //    printf("v = %d inEdges[v] = %zu\n", v, inEdges[v]->size());
  //    cnt += inEdges[v]->size() * inEdges[v]->size();
  //  }
  //printf("cnt = %d\n", cnt);
  //fclose(file);
 
  float* inputZC = (float*) malloc(nv * HIDDEN_SIZE * sizeof(float));
  memset(inputZC, 0, nv * HIDDEN_SIZE * sizeof(float));
  for (v = 0; v < nv; v++)
    if (inEdges.find(v) != inEdges.end())
      inputZC[v * HIDDEN_SIZE + inEdges[v]->size() % HIDDEN_SIZE] = 1.0f;
    else
      inputZC[v * HIDDEN_SIZE] = 1.0f;
    
  float* inputFB;
  checkCUDA(cudaMalloc(&inputFB, nv * HIDDEN_SIZE * sizeof(float)));
  checkCUDA(cudaMemcpy(inputFB, inputZC, nv * HIDDEN_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));

  // Optimize Computation Graph
  std::map<V_ID, std::set<V_ID>*> optInEdges;
  std::vector<std::pair<V_ID, V_ID> > optRanges;
  V_ID newNv;
  transfer_graph(inEdges, optInEdges, optRanges,
                 nv, ne, maxDepth, maxWidth, newNv);
  GNNModel model(handle);
  model.set_dep_graph(nv, newNv, nv, optInEdges, optRanges);
  //std::vector<std::pair<V_ID, V_ID> > ranges;
  //model.set_dep_graph(nv, nv, nv, inEdges, ranges);
  model.load_node_label(nv, nodeLabelFile);

  // Init adam optimizer
  AdamOpt adam;
  adam.alpha = learningRate;

  std::vector<Layer*> layers;
  for (int i = 0; i < NUM_LAYERS; i++) {
    float* inputPtr = (i == 0) ? inputFB : layers[i-1]->outputPtr;
    float* inputGradPtr = (i == 0) ? NULL : layers[i-1]->outputGradPtr;
    layers.push_back(new GNNLayer(&model, inputPtr, inputGradPtr,
                             HIDDEN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE,
                             ACT_MODE_RELU, AGG_MODE_MEAN_POOLING));
  }
  //layers.push_back(new NCLayer(&model, inputFB, NULL, HIDDEN_SIZE, model.numClass));
  layers.push_back(new NCLayer(&model, layers[layers.size() - 1]->outputPtr,
                               layers[layers.size() - 1]->outputGradPtr,
                               HIDDEN_SIZE, model.numClass));
  layers.push_back(new SMLayer(&model, layers[layers.size() - 1]->outputPtr,
                               layers[layers.size() - 1]->outputGradPtr,
                               nv, model.numClass));

  cudaEvent_t startEvent, endEvent;
  checkCUDA(cudaEventCreate(&startEvent));
  checkCUDA(cudaEventCreate(&endEvent));
  checkCUDA(cudaDeviceSynchronize());
  checkCUDA(cudaEventRecord(startEvent));
  for (int iter = 0; iter < epochs; iter ++) {
    adam.next_epoch();
    for (int i = 0; i < layers.size(); i++) {
      layers[i]->forward();
    }
    for (int i = layers.size() - 1; i >= 0; i--) {
      layers[i]->backward();
      layers[i]->update(adam);
    }
  }
  checkCUDA(cudaEventRecord(endEvent));
  checkCUDA(cudaEventSynchronize(endEvent));
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, startEvent, endEvent);
  printf("EXECUTION TIME = %.4lfms\n", milliseconds);
}

GNNModel::GNNModel(Handler _handle)
: handle(_handle) {}

void GNNModel::set_graph(Graph& graph, V_ID nvSrc, V_ID nvNewSrc, V_ID nvDst,
                         std::map<V_ID, std::set<V_ID>* >& inEdges,
                         std::vector<std::pair<V_ID, V_ID> >& ranges)
{
  graph.nvSrc = nvSrc;
  graph.nvNewSrc = nvNewSrc;
  graph.nvDst = nvDst;
  graph.ne = 0;
  graph.ranges = ranges;
  std::map<V_ID, std::set<V_ID>* > outEdges;
  for (V_ID v = 0; v < nvNewSrc; v++)
    if (inEdges.find(v) != inEdges.end())
      graph.ne += inEdges[v]->size();
  NodeStruct *rowPtrZC, *inRowPtrFB, *outRowPtrFB;
  EdgeStruct *colIdxZC, *inColIdxFB, *outColIdxFB;
  V_ID *inDegZC, *inDegFB;
  rowPtrZC = (NodeStruct*) malloc(graph.nvNewSrc * sizeof(NodeStruct));
  colIdxZC = (EdgeStruct*) malloc(graph.ne * sizeof(EdgeStruct));
  inDegZC = (V_ID*) malloc(graph.nvNewSrc * sizeof(V_ID));
  // Step 1: compute in-degree
  for (V_ID v = nvSrc; v < nvNewSrc; v++) {
    inDegZC[v] = 0;
    assert(inEdges.find(v) != inEdges.end());
    std::set<V_ID>::const_iterator it;
    for (it = inEdges[v]->begin(); it != inEdges[v]->end(); it++) {
      inDegZC[v] += *it < nvSrc ? 1 : inDegZC[*it];
    }
  }
  for (V_ID v = 0; v < nvSrc; v++) {
    inDegZC[v] = 0;
    if (inEdges.find(v) != inEdges.end()) {
      std::set<V_ID>::const_iterator first = inEdges[v]->begin();
      std::set<V_ID>::const_iterator last = inEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++)
        inDegZC[v] += *it < nvSrc ? 1 : inDegZC[*it];
    }
  }
  // Step 2: construct in edges;
  E_ID count = 0;
  for (V_ID v = 0; v < nvNewSrc; v++) {
    if (inEdges.find(v) != inEdges.end()) {
      std::set<V_ID>::const_iterator first = inEdges[v]->begin();
      std::set<V_ID>::const_iterator last = inEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++) {
        colIdxZC[count].src = *it;
        colIdxZC[count].dst = v;
        count ++;
        if (outEdges.find(*it) == outEdges.end())
          outEdges[*it] = new std::set<V_ID>();
        outEdges[*it]->insert(v);
      }
    }
    rowPtrZC[v].index = count;
  }
  checkCUDA(cudaMalloc(&inRowPtrFB, graph.nvNewSrc * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&inColIdxFB, graph.ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMalloc(&inDegFB, graph.nvNewSrc * sizeof(V_ID)));
  checkCUDA(cudaMemcpy(inRowPtrFB, rowPtrZC, graph.nvNewSrc * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(inColIdxFB, colIdxZC, graph.ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(inDegFB, inDegZC, graph.nvNewSrc * sizeof(V_ID),
                       cudaMemcpyHostToDevice));
  graph.inRowPtr = inRowPtrFB;
  graph.inColIdx = inColIdxFB;
  graph.inDeg = inDegFB;
  // Step 3: construct out edges
  count = 0;
  for (V_ID v = 0; v < nvNewSrc; v++) {
    if (outEdges.find(v) != outEdges.end()) {
      std::set<V_ID>::const_iterator first = outEdges[v]->begin();
      std::set<V_ID>::const_iterator last = outEdges[v]->end();
      std::set<V_ID>::const_iterator it = first;
      for (it = first; it != last; it++) {
        colIdxZC[count].src = *it;
        colIdxZC[count].dst = v;
        count ++;
      }
    }
    rowPtrZC[v].index = count;
  }
  checkCUDA(cudaMalloc(&outRowPtrFB, graph.nvNewSrc * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&outColIdxFB, graph.ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMemcpy(outRowPtrFB, rowPtrZC, graph.nvNewSrc * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(outColIdxFB, colIdxZC, graph.ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  graph.outRowPtr = outRowPtrFB;
  graph.outColIdx = outColIdxFB;
  // Step 3: free resources
  free(rowPtrZC);
  free(colIdxZC);
  free(inDegZC);
}

void GNNModel::set_dep_graph(V_ID nvSrc, V_ID nvNewSrc, V_ID nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList,
         std::vector<std::pair<V_ID, V_ID> >& ranges)
{
  set_graph(depGraph, nvSrc, nvNewSrc, nvDst, edgeList, ranges);
  printf("Add normal in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         depGraph.nvSrc, depGraph.nvDst, depGraph.ne);
}

void GNNModel::set_hyper_graph(int nvSrc, int nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  assert(nvSrc >= nvDst);
  std::vector<std::pair<V_ID, V_ID> > ranges;
  set_graph(hyGraph, nvSrc, nvSrc, nvDst, edgeList, ranges);
  printf("Add hyper in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         hyGraph.nvSrc, hyGraph.nvDst, hyGraph.ne);
}

void GNNModel::load_node_label(int nv, std::string filename)
{
  FILE* file = fopen(filename.c_str(), "r");
  int* labelsZC = (int*) malloc(nv * sizeof(int));
  V_ID cnt = 0;
  int u;
  numClass = 0;
  while (fscanf(file, "%d", &u) != EOF) {
    labelsZC[cnt ++] = u;
    if (u >= numClass) numClass = u + 1;
  }
  printf("numClass = %d\n", numClass);
  assert(cnt == nv);
  checkCUDA(cudaMalloc(&labels, nv * sizeof(int)));
  checkCUDA(cudaMemcpy(labels, labelsZC, nv * sizeof(int),
                       cudaMemcpyHostToDevice));
  free(labelsZC);
  fclose(file);
}

Layer::Layer(GNNModel* _model, float* _inputPtr, float* _inputGradPtr)
: model(_model), inputPtr(_inputPtr), inputGradPtr(_inputGradPtr)
{}

void AdamOpt::next_epoch(void)
{
  beta1_t *= beta1;
  beta2_t *= beta2;
  alpha_t = alpha * sqrt(1 - beta2_t) / (1 - beta1_t);
}
