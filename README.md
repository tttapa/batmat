# batmat

Fast linear algebra routines for batches of small matrices.

## Development installation

Using GCC:
```sh
for bt in Debug Release; do for tf in True False; do
    conan install . --build=missing -pr scripts/dev/profiles/desktop -c tools.cmake.cmaketoolchain:generator=Ninja -c tools.build:skip_test=$tf -s build_type=$bt -c tools.build:jobs=4 -o \*:with_openmp=True -o \*:with_openblas=False -o \*:with_mkl=True -o \*:with_benchmarks=True -o \*:with_tracing=True
done; done
```

Using Clang:
```sh
for bt in Debug Release; do for tf in True False; do
    conan install . --build=missing -pr scripts/dev/profiles/desktop-clang20 -c tools.cmake.cmaketoolchain:generator=Ninja -c tools.build:skip_test=$tf -s build_type=$bt -c tools.build:jobs=4 -o \*:with_openmp=False -o \*:with_openblas=False -o \*:with_mkl=True -o \*:with_benchmarks=True -o \*:with_tracing=True -o \&:with_gsi_hpc_simd=True
done; done
```
