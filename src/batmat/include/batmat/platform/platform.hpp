#pragma once

#if defined(__AVX512F__)
#include "avx-512.hpp"
#elif defined(__AVX2__)
#include "avx2.hpp"
#elif defined(__ARM_NEON)
#include "neon.hpp"
#else
#include "generic.hpp"
#endif
