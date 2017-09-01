# config
CUDA=1
CUDNN=0
USE_SSE2=1
CUDA_PATH=/usr/local/cuda
CUDNN_PATH=/usr/local/cuda
ARCH= --gpu-architecture=compute_50 --gpu-code=compute_50

DEBUG=0
CC=gcc
NVCC=$(CUDA_PATH)/bin/nvcc
OPTS=-O3
LDFLAGS=-lm -lrt
CFLAGS=-Wall -Wfatal-errors 
DELIVERY_BIN=bin/bcnn-cl
DELIVERY_LIB=lib/libbcnn.a
DELIVERY_EXAMPLE=bin/examples/mnist-example

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS) -Iinc

ifeq ($(CUDA), 1) 
CFLAGS+= -DBCNN_USE_CUDA -I$(CUDA_PATH)/include/
LDFLAGS+= -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1)
CFLAGS+= -DBCNN_USE_CUDNN -I$(CUDNN_PATH)/include/
LDFLAGS+= -L$(CUDNN_PATH)/lib64 -lcudnn 
endif

ifeq ($(USE_SSE2), 1)
CFLAGS+= -DBCNN_USE_SSE2 -msse2
endif

# bip library
BIP_PATH=bip
CFLAGS += -DBIP_USE_STB_IMAGE -I$(BIP_PATH)/inc
LIB_DEP += $(BIP_PATH)/libbip.a
LDFLAGS += -L$(BIP_PATH)

# bh headers
BH_PATH=bh
CFLAGS += -I$(BH_PATH)/inc

SRC = $(wildcard src/*.c)
OBJ = $(patsubst src/%.c, build/%.o, $(SRC))
SRC_CUDA = $(wildcard src/*.cu)
OBJ_CUDA = $(patsubst src/%.cu, build/%_gpu.o, $(SRC_CUDA))
CL_OBJ = build/bcnn_cl.o
SRC_PACK = tools/pack_img/pack_img.c
PACK_OBJ = build/tools/pack_img.o
SRC_MNISTEXAMPLE = examples/mnist/mnist_example.c
OBJ_MNISTEXAMPLE = build/examples/mnist_example.o

ALL_OBJ = $(OBJ)
ifeq ($(CUDA), 1)
ALL_OBJ += $(OBJ_CUDA)
endif
ALL_DEP = $(filter-out build/bcnn_cl.o, $(ALL_OBJ)) $(LIB_DEP)

all: clean $(DELIVERY_LIB) $(DELIVERY_BIN) $(DELIVERY_EXAMPLE)

$(BIP_PATH)/libbip.a: 
	cd $(BIP_PATH); make; cd ../

build/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/%_gpu.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(ARCH) --compiler-options "$(CFLAGS)" -c $< -o $@
	
$(DELIVERY_LIB): $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

$(DELIVERY_BIN): $(CL_OBJ) $(ALL_DEP)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

$(PACK_OBJ): $(SRC_PACK)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

pack-img: $(PACK_OBJ) $(ALL_DEP)
	$(CC) $(CFLAGS) -o $@ $(filter %.o %.a, $^) $(LDFLAGS)
	
$(OBJ_MNISTEXAMPLE): $(SRC_MNISTEXAMPLE)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(DELIVERY_EXAMPLE): $(OBJ_MNISTEXAMPLE) $(ALL_DEP)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(OBJ) $(OBJ_CUDA) $(DELIVERY_BIN) $(BIP_PATH)/libbip.a $(DELIVERY_LIB) $(PACK_OBJ) pack-img $(OBJ_MNISTEXAMPLE) $(DELIVERY_EXAMPLE)
	cd $(BIP_PATH); make clean; cd ../
