#include <benchmark/benchmark.h>
#include <batmat-version.h>
#include <format>
#include <string>

#include <batmat/config.hpp>
#include <batmat/dtypes.hpp>
#include <batmat/openmp.h>
#include <guanaqo/blas/config.hpp>
#include <guanaqo/demangled-typename.hpp>
#include <guanaqo/pcm/counters.hpp>
#include <guanaqo/perfetto/trace.hpp>
#include <guanaqo/string-util.hpp>
#include <guanaqo/stringify.h>
#include <guanaqo-version.h>

#if GUANAQO_WITH_MKL
#include <mkl.h>
#elif GUANAQO_WITH_OPENBLAS
#include <openblas_config.h>
#endif

#ifdef BATMAT_WITH_EIGEN
#include <Eigen/Version>
#endif

void register_context() {
    benchmark::AddCustomContext("batmat_build_time", batmat_build_time);
    benchmark::AddCustomContext("batmat_commit_hash", batmat_commit_hash);
    benchmark::AddCustomContext("real_t", guanaqo::demangled_typename(typeid(batmat::real_t)));
    benchmark::AddCustomContext("index_t", guanaqo::demangled_typename(typeid(batmat::index_t)));
#if BATMAT_WITH_OPENMP
    auto places = std::getenv("OMP_PLACES");
    benchmark::AddCustomContext("OMP_NUM_THREADS", std::to_string(omp_get_max_threads()));
    benchmark::AddCustomContext("OMP_PLACES", places ? places : "");
#endif
#if GUANAQO_WITH_MKL
    MKLVersion Version;
    mkl_get_version(&Version);
    benchmark::AddCustomContext("blas_library", "MKL");
    benchmark::AddCustomContext("blas_version",
                                std::format("{}.{}.{} ({}) for {}: {}", Version.MajorVersion,
                                            Version.MinorVersion, Version.UpdateVersion,
                                            Version.Build, Version.Platform, Version.Processor));
#elif GUANAQO_WITH_OPENBLAS
    benchmark::AddCustomContext("blas_library", "OpenBLAS");
    benchmark::AddCustomContext("blas_version", OPENBLAS_VERSION);
#endif
#if defined(__INTEL_LLVM_COMPILER)
    benchmark::AddCustomContext("compiler", "intel-llvm");
    benchmark::AddCustomContext("compiler_version", __VERSION__);
#elif defined(__clang__)
    benchmark::AddCustomContext("compiler", "clang");
    benchmark::AddCustomContext("compiler_version", __VERSION__);
#elif defined(__GNUC__)
    benchmark::AddCustomContext("compiler", "gcc");
    benchmark::AddCustomContext("compiler_version", __VERSION__);
#elif defined(_MSC_VER)
    benchmark::AddCustomContext("compiler", "msvc");
    benchmark::AddCustomContext("compiler_version", GUANAQO_STRINGIFY(_MSC_FULL_VER));
#endif
#if defined(__AVX512F__)
    benchmark::AddCustomContext("arch", "avx512f");
#elif defined(__AVX2__)
    benchmark::AddCustomContext("arch", "avx2");
#elif defined(__AVX__)
    benchmark::AddCustomContext("arch", "avx");
#elif defined(__SSE3__)
    benchmark::AddCustomContext("arch", "sse3");
#elif defined(__ARM_NEON)
    benchmark::AddCustomContext("arch", "neon");
#endif
#if BATMAT_WITH_GSI_HPC_SIMD
    benchmark::AddCustomContext("simd_library", "GSI-HPC/simd");
#else
    benchmark::AddCustomContext("simd_library", "libstdc++ <experimental/simd>");
#endif
#if BATMAT_WITH_EIGEN
    benchmark::AddCustomContext("eigen_version", EIGEN_VERSION_STRING);
#endif
}

int main(int argc, char **argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    register_context();
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
}
