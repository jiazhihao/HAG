all: gnn

INCFLAGS = -Icub/

CXXFLAGS = -O2 -arch=compute_60 -code=sm_60 -std=c++11

LDFLAGS = -lcudnn -lcublas -lcurand

gnn: gnn.cc gnn_kernel.cu
	nvcc -o $@ $(INCFLAGS) $(CXXFLAGS) $^ $(LDFLAGS)
