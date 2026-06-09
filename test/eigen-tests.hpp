#pragma once

#if defined(__GNUC__) && !defined(__clang__)
#define BATMAT_PRAGMA_GCC_OPTIMIZE_O3_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"O3\")")
#define BATMAT_PRAGMA_GCC_OPTIMIZE_O3_END _Pragma("GCC pop_options")
#define BATMAT_ATTR_GNU_OPTIMIZE_O3 [[gnu::optimize("O3")]]
#else
#define BATMAT_PRAGMA_GCC_OPTIMIZE_O3_BEGIN
#define BATMAT_PRAGMA_GCC_OPTIMIZE_O3_END
#define BATMAT_ATTR_GNU_OPTIMIZE_O3
#endif

BATMAT_PRAGMA_GCC_OPTIMIZE_O3_BEGIN
#include <Eigen/Core>
BATMAT_PRAGMA_GCC_OPTIMIZE_O3_END
