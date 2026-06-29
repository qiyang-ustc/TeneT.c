PREFIX ?= ../../deps
PREFIX_ABS := $(abspath $(PREFIX))
CXX ?= c++
NVCC ?= nvcc
JULIA ?= julia
CUDA_ARCH ?= sm_90
BLAS_INTEGER ?= 64
UNAME_S := $(shell uname -s)
CUDA_HOME_DIR ?= $(shell dirname $(dirname $(shell command -v $(NVCC) 2>/dev/null || echo $(NVCC))))
CUDA_LIBDIR ?= $(if $(wildcard $(CUDA_HOME_DIR)/lib64),$(CUDA_HOME_DIR)/lib64,$(CUDA_HOME_DIR)/lib)
JULIA_BLAS_LIB := $(shell $(JULIA) --startup-file=no -e 'using LinearAlgebra; using Libdl; print(Libdl.dlpath(Base.libblas_name))' 2>/dev/null)
JULIA_BLAS_LIBDIR := $(patsubst %/,%,$(dir $(JULIA_BLAS_LIB)))

CPU_RPATH_DIRS = $(strip $(JULIA_LIBDIR) $(PREFIX_ABS))
CPU_RPATH_FLAGS = $(foreach dir,$(CPU_RPATH_DIRS),-Wl,-rpath,$(dir))
CUDA_RPATH_DIRS = $(strip $(JULIA_LIBDIR) $(PREFIX_ABS) $(CUDA_LIBDIR))
CUDA_RPATH_FLAGS = $(foreach dir,$(CUDA_RPATH_DIRS),-Xlinker -rpath -Xlinker $(dir))

ifeq ($(UNAME_S),Darwin)
DL_EXT ?= dylib
SHARED_FLAGS ?= -dynamiclib
JULIA_LIBDIR ?= $(JULIA_BLAS_LIBDIR)
BLAS_LIBS ?= $(JULIA_BLAS_LIB)
RPATH_FLAGS ?= $(CPU_RPATH_FLAGS)
NVCC_RPATH_FLAGS ?= $(CUDA_RPATH_FLAGS)
else
DL_EXT ?= so
SHARED_FLAGS ?= -shared
JULIA_LIBDIR ?= $(JULIA_BLAS_LIBDIR)
BLAS_LIBS ?= $(JULIA_BLAS_LIB)
RPATH_FLAGS ?= $(CPU_RPATH_FLAGS)
NVCC_RPATH_FLAGS ?= $(CUDA_RPATH_FLAGS)
endif

ifeq ($(strip $(BLAS_LIBS)),)
BLAS_LIBS := $(JULIA_BLAS_LIB)
endif
ifeq ($(strip $(JULIA_LIBDIR)),)
JULIA_LIBDIR := $(JULIA_BLAS_LIBDIR)
endif

ifeq ($(filter 32 64,$(BLAS_INTEGER)),)
$(error BLAS_INTEGER must be 32 or 64)
endif

CXXFLAGS ?= -O3 -std=c++17 -fPIC -Wall -Wextra
ifeq ($(BLAS_INTEGER),64)
CXXFLAGS += -DTENET_NATIVE_USE_BLAS64
endif
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler=-fPIC -arch=$(CUDA_ARCH)
ifeq ($(BLAS_INTEGER),64)
NVCCFLAGS += -DTENET_NATIVE_USE_BLAS64
endif
NVCC_LDFLAGS ?=
NVCC_BLAS_LIBS ?= -Xlinker $(BLAS_LIBS)

NATIVE_ARNOLDI_CPU_LIB := $(PREFIX_ABS)/libtenet_native_arnoldi.$(DL_EXT)
NATIVE_ARNOLDI_CUDA_LIB := $(PREFIX_ABS)/libtenet_native_arnoldi_cuda.$(DL_EXT)
NATIVE_ISING_CPU_DRIVER := $(PREFIX_ABS)/tenet_native_ising_cpu_driver
NATIVE_ISING_CUDA_DRIVER := $(PREFIX_ABS)/tenet_native_ising_cuda_driver
NATIVE_TFISING_CPU_DRIVER := $(PREFIX_ABS)/tenet_native_tfising_cpu_driver
NATIVE_SHARED_CPU_HEADER := tenet_native_shared_two_layer.hpp

.PHONY: all native-arnoldi-cpu native-arnoldi-cuda native-ising-cpu-driver native-ising-cuda-driver native-tfising-cpu-driver drivers clean

all: native-arnoldi-cpu

native-arnoldi-cpu: $(NATIVE_ARNOLDI_CPU_LIB)

native-arnoldi-cuda: $(NATIVE_ARNOLDI_CUDA_LIB)

native-ising-cpu-driver: $(NATIVE_ISING_CPU_DRIVER)

native-ising-cuda-driver: $(NATIVE_ISING_CUDA_DRIVER)

native-tfising-cpu-driver: $(NATIVE_TFISING_CPU_DRIVER)

drivers: native-ising-cpu-driver native-tfising-cpu-driver

$(NATIVE_ARNOLDI_CPU_LIB): tenet_native_arnoldi_cpu.cpp tenet_native_arnoldi.h krylov_core.hpp $(NATIVE_SHARED_CPU_HEADER)
	mkdir -p $(PREFIX_ABS)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -o $@ tenet_native_arnoldi_cpu.cpp $(BLAS_LIBS) $(RPATH_FLAGS)

$(NATIVE_ISING_CPU_DRIVER): tenet_native_ising_cpu_driver.cpp tenet_native_arnoldi.h $(NATIVE_ARNOLDI_CPU_LIB)
	mkdir -p $(PREFIX_ABS)
	$(CXX) $(CXXFLAGS) -o $@ tenet_native_ising_cpu_driver.cpp -L$(PREFIX_ABS) -ltenet_native_arnoldi $(RPATH_FLAGS)

$(NATIVE_TFISING_CPU_DRIVER): tenet_native_tfising_cpu_driver.cpp tenet_native_arnoldi.h $(NATIVE_ARNOLDI_CPU_LIB)
	mkdir -p $(PREFIX_ABS)
	$(CXX) $(CXXFLAGS) -o $@ tenet_native_tfising_cpu_driver.cpp -L$(PREFIX_ABS) -ltenet_native_arnoldi $(BLAS_LIBS) $(RPATH_FLAGS)

$(NATIVE_ARNOLDI_CUDA_LIB): tenet_native_arnoldi_cuda.cu tenet_native_arnoldi.h
	mkdir -p $(PREFIX_ABS)
	$(NVCC) $(NVCCFLAGS) --shared --cudart=shared -o $@ tenet_native_arnoldi_cuda.cu -lcublas $(NVCC_BLAS_LIBS) $(NVCC_RPATH_FLAGS) $(NVCC_LDFLAGS)

$(NATIVE_ISING_CUDA_DRIVER): tenet_native_ising_cuda_driver.cu tenet_native_arnoldi.h $(NATIVE_ARNOLDI_CUDA_LIB) $(NATIVE_ARNOLDI_CPU_LIB)
	mkdir -p $(PREFIX_ABS)
	$(NVCC) $(NVCCFLAGS) -o $@ tenet_native_ising_cuda_driver.cu -L$(PREFIX_ABS) -ltenet_native_arnoldi_cuda -ltenet_native_arnoldi -lcublas $(NVCC_BLAS_LIBS) $(NVCC_RPATH_FLAGS) $(NVCC_LDFLAGS) -Xlinker -rpath -Xlinker $(PREFIX_ABS)

clean:
	rm -f $(NATIVE_ARNOLDI_CPU_LIB) $(NATIVE_ARNOLDI_CUDA_LIB) $(NATIVE_ISING_CPU_DRIVER) $(NATIVE_ISING_CUDA_DRIVER) $(NATIVE_TFISING_CPU_DRIVER)
