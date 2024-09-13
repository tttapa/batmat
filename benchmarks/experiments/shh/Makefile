VERSION := v1

BUILD_DIR := ../../../build
BUILD_TYPE := Release
SHH_EXE := $(BUILD_DIR)/benchmarks/$(BUILD_TYPE)/benchmark-shh

all:
	@echo "Please specify a target"
.PHONY: all

results/n8-serial-$(VERSION).json: $(SHH_EXE)
	OMP_NUM_THREADS=1 $(SHH_EXE) \
	    --benchmark_filter='(blas|shh<.*>)/' \
	    --fix_n --n=8 --m=16 \
	    --benchmark_min_time=0.02s --benchmark_repetitions=5 \
	    --benchmark_enable_random_interleaving \
	    --benchmark_out=$@
n8: results/n8-serial-$(VERSION).json
.PHONY: n8

results/n64-serial-$(VERSION).json: $(SHH_EXE)
	OMP_NUM_THREADS=1 $(SHH_EXE) \
	    --benchmark_filter='(blas|shh<.*>)/' \
	    --fix_n --n=64 --m=128 \
	    --benchmark_min_time=0.02s --benchmark_repetitions=5 \
	    --benchmark_enable_random_interleaving \
	    --benchmark_out=$@
n64: results/n64-serial-$(VERSION).json
.PHONY: n64

results/n512-serial-$(VERSION).json: $(SHH_EXE)
	OMP_NUM_THREADS=1 $(SHH_EXE) \
	    --benchmark_filter='(blas|shh<.*>)/' \
	    --fix_n --n=512 --m=1024 \
	    --benchmark_min_time=0.02s --benchmark_repetitions=5 \
	    --benchmark_enable_random_interleaving \
	    --benchmark_out=$@
n512: results/n512-serial-$(VERSION).json
.PHONY: n512

results/n1024-serial-$(VERSION).json: $(SHH_EXE)
	OMP_NUM_THREADS=1 $(SHH_EXE) \
	    --benchmark_filter='(blas|shh<.*>)/' \
	    --fix_n --n=1024 --m=2048 \
	    --benchmark_min_time=0.02s --benchmark_repetitions=5 \
	    --benchmark_enable_random_interleaving \
	    --benchmark_out=$@
n1024: results/n1024-serial-$(VERSION).json
.PHONY: n1024
