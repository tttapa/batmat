```sh
OMP_NUM_THREADS=8 ./build/benchmarks/ocp-solve/Release/benchmark-ocp-solve \
    --benchmark_out=ocp-solve.json --benchmark_repetitions=3 --benchmark_min_time=0.05s \
    --N=31 --vary-nu --step=4 --nx=40 --nu=40 --ny=2
```
