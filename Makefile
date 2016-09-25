# config
CUDA=1
CUDNN=0
DEBUG=0
USE_SSE2=1
CUDA_PATH=/usr/local/cuda
ARCH= --gpu-architecture=compute_20 --gpu-code=compute_20

CC=gcc
NVCC=$(CUDA_PATH)/bin/nvcc
OPTS=-O3
LDFLAGS=-lm 
CFLAGS=-Wall -Wfatal-errors 
BIN=bcnn-cl

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS) -Iinc

ifeq ($(CUDA), 1) 
CFLAGS+= -DBCNN_USE_CUDA -I$(CUDA_PATH)/include/
LDFLAGS+= -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
LDFLAGS+= -L$(CUDNN_PATH) -lcudnn -I$(CUDNN_PATH)
CFLAGS+= -DBCNN_USE_CUDNN
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

ALL_OBJ = $(OBJ)
ifeq ($(CUDA), 1)
ALL_OBJ += $(OBJ_CUDA)
endif
ALL_DEP = $(filter-out build/bcnn_cl.o, $(ALL_OBJ)) $(LIB_DEP)

all: clean lib/libbcnn.a $(BIN)

$(BIP_PATH)/libbip.a: 
	cd $(BIP_PATH); make; cd ../

build/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

build/%_gpu.o: src/%.cu
	@mkdir -p $(@D)
	$(NVCC) $(ARCH) --compiler-options "$(CFLAGS)" -c $< -o $@
	
lib/libbcnn.a: $(ALL_DEP)
	@mkdir -p $(@D)
	ar crv $@ $(filter %.o, $?)

$(BIN): $(CL_OBJ) $(ALL_DEP)
	$(CC) $(CFLAGS) -o $@ $(filter %.o %.a, $^) $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(OBJ) $(OBJ_CUDA) $(BIN) $(BIP_PATH)/libbip.a lib/libbcnn.a
	cd $(BIP_PATH); make clean; cd ../