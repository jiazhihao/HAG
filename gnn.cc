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
  parse_input_args(argv, argc, graphFile, hyGraphFile,
                   nodeLabelFile, graphLabelFile,
                   learningRate, epochs);
  Handler handle;
  //FILE* file = fopen("BZR_MD/BZR_MD_A.txt", "r");
  FILE* file = fopen(graphFile.c_str(), "r");
  V_ID u, v;
  V_ID nv = 0;
  std::map<V_ID, std::set<V_ID>* > inEdges, outEdges;

  while (fscanf(file, "%d, %d", &u, &v) != EOF) {
    // shift node indices by 1 to make them 0-indexed
    u --;
    // shift node indices by 1 to make them 0-indexed
    v --;
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

  AdamOpt adam;
  adam.alpha = learningRate;
  GNNModel model(handle);
  model.set_in_graph(nv, nv, inEdges);
  model.set_out_graph(nv, nv, outEdges);
  model.load_node_label(nv, nodeLabelFile);

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
}

GNNModel::GNNModel(Handler _handle)
: handle(_handle) {}

void GNNModel::set_graph(Graph& graph, int nvSrc, int nvDst,
                         std::map<V_ID, std::set<V_ID>* >& inEdges)
{
  graph.nvSrc = nvSrc;
  graph.nvDst = nvDst;
  graph.ne = 0;
  for (int v = 0; v < nvDst; v++)
    if (inEdges.find(v) != inEdges.end())
      graph.ne += inEdges[v]->size();
  NodeStruct *rowPtrZC, *rowPtrFB;
  EdgeStruct *colIdxZC, *colIdxFB;
  V_ID *inDegZC, *inDegFB;
  rowPtrZC = (NodeStruct*) malloc(graph.nvDst * sizeof(NodeStruct));
  colIdxZC = (EdgeStruct*) malloc(graph.ne * sizeof(EdgeStruct));
  inDegZC = (V_ID*) malloc(graph.nvDst * sizeof(V_ID));
  E_ID count = 0;
  for (int v = 0; v < nvDst; v++) {
    if (inEdges.find(v) != inEdges.end()) {
      inDegZC[v] = inEdges[v]->size();
      std::set<V_ID>::const_iterator it;
      for (it = inEdges[v]->begin(); it != inEdges[v]->end(); it++) {
        colIdxZC[count].src = *it;
        colIdxZC[count].dst = v;
        count ++;
      }
    } else {
      inDegZC[v] = 0;
    }
    rowPtrZC[v].index = count;
  }
  checkCUDA(cudaMalloc(&rowPtrFB, graph.nvDst * sizeof(NodeStruct)));
  checkCUDA(cudaMalloc(&colIdxFB, graph.ne * sizeof(EdgeStruct)));
  checkCUDA(cudaMalloc(&inDegFB, graph.nvDst * sizeof(V_ID)));
  checkCUDA(cudaMemcpy(rowPtrFB, rowPtrZC, graph.nvDst * sizeof(NodeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(colIdxFB, colIdxZC, graph.ne * sizeof(EdgeStruct),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaMemcpy(inDegFB, inDegZC, graph.nvDst * sizeof(V_ID),
                       cudaMemcpyHostToDevice));
  free(rowPtrZC);
  free(colIdxZC);
  free(inDegZC);
  graph.rowPtr = rowPtrFB;
  graph.colIdx = colIdxFB;
  graph.inDeg = inDegFB;
}

void GNNModel::set_in_graph(int nvSrc, int nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(inGraph, nvSrc, nvDst, edgeList);
  printf("Add normal in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         inGraph.nvSrc, inGraph.nvDst, inGraph.ne);
}

void GNNModel::set_out_graph(int nvSrc, int nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(outGraph, nvSrc, nvDst, edgeList);
  printf("Add normal out-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         outGraph.nvSrc, outGraph.nvDst, outGraph.ne);
}

void GNNModel::set_hyper_in_graph(int nvSrc, int nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(hyInGraph, nvSrc, nvDst, edgeList);
  printf("Add hyper in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         hyInGraph.nvSrc, hyInGraph.nvDst, hyInGraph.ne);
}

void GNNModel::set_hyper_out_graph(int nvSrc, int nvDst,
         std::map<V_ID, std::set<V_ID>* >& edgeList)
{
  set_graph(hyOutGraph, nvSrc, nvDst, edgeList);
  printf("Add hyper in-edge graph: nvSrc(%d) nvDst(%d) ne(%d)\n",
         hyOutGraph.nvSrc, hyOutGraph.nvDst, hyOutGraph.ne);
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
