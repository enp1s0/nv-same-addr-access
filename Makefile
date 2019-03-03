NVCC=nvcc
NVCCFLAGS=-std=c++14 -arch=sm_60
TARGET=nvmem-test

$(TARGET):main.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<
