#pragma once

#if defined(_MSC_VER) // For MSVC
    #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__) // For GCC/Clang
    #define FORCE_INLINE inline __attribute__((always_inline))
#else
    #define FORCE_INLINE inline
    #warning "Unknown compiler, inlining is not guaranteed"
#endif