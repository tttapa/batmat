# batmat

Fast linear algebra routines for batches of small matrices.

## Development installation

```sh
for bt in Debug Release; do for tf in True False; do
    conan install . --build=missing -pr scripts/dev/profiles/desktop -c tools.cmake.cmaketoolchain:generator=Ninja\ Multi-Config -c tools.build:skip_test=$tf -s build_type=$bt -o \*:with_openmp=True -o \*:with_openblas=False -o \*:with_mkl=True -o \*:with_benchmarks=True -o \*:with_tracing=True
done; done
```
