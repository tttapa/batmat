```sh
OMP_NUM_THREADS=8 taskset -c 0,1,2,3,4,5,6,7 \
./build/benchmarks/ocp-solve/Release/benchmark-ocp-solve \
    --benchmark_out=ocp-solve-nu.json --benchmark_repetitions=5 --benchmark_min_time=0.05s \
    --N=31 --vary-nu --step=4 --nx=40 --nu=40 --ny=8
OMP_NUM_THREADS=8 taskset -c 0,1,2,3,4,5,6,7 \
./build/benchmarks/ocp-solve/Release/benchmark-ocp-solve \
    --benchmark_out=ocp-solve-nx.json --benchmark_repetitions=5 --benchmark_min_time=0.05s \
    --N=31 --vary-nx-frac --step=4 --nx=100 --nu=25 --ny=8
OMP_NUM_THREADS=8 taskset -c 0,1,2,3,4,5,6,7 \
./build/benchmarks/ocp-solve/Release/benchmark-ocp-solve \
    --benchmark_filter='bm_factor_riccati_blasfeo.*' \
    --benchmark_out=ocp-solve-N-blasfeo.json --benchmark_repetitions=5 --benchmark_min_time=0.05s \
    --N=256 --vary-N-pow-2 --step=1 --nx=40 --nu=20 --ny=0
```
