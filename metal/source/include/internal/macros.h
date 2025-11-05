#pragma once

#ifdef mindfour_ARCH_X86_64
#if mindfour_ARCH_X86_64 != 0 && mindfour_ARCH_X86_64 != 1
#error "Invalid mindfour_ARCH_X86_64 value: must be either 0 or 1"
#endif
#else
#if defined(__x86_64__) || defined(_M_X64) && !defined(_M_ARM64EC)
#define mindfour_ARCH_X86_64 1
#else
#define mindfour_ARCH_X86_64 0
#endif
#endif

#ifdef mindfour_ARCH_ARM64
#if mindfour_ARCH_ARM64 != 0 && mindfour_ARCH_ARM64 != 1
#error "Invalid mindfour_ARCH_ARM64 value: must be either 0 or 1"
#endif
#else
#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)
#define mindfour_ARCH_ARM64 1
#else
#define mindfour_ARCH_ARM64 0
#endif
#endif

#if mindfour_ARCH_X86_64 + mindfour_ARCH_ARM64 == 0
#error "Unsupported architecture: neither x86-64 nor ARM64 detected"
#elif mindfour_ARCH_X86_64 + mindfour_ARCH_ARM64 != 1
#error "Inconsistent architecture detection: both x86-64 and ARM64 detection macros are specified"
#endif

/***** Compiler portability macros *****/

#ifndef mindfour_LIKELY
#if defined(__GNUC__)
#define mindfour_LIKELY(condition) (__builtin_expect(!!(condition), 1))
#else
#define mindfour_LIKELY(condition) (!!(condition))
#endif
#endif

#ifndef mindfour_UNLIKELY
#if defined(__GNUC__)
#define mindfour_UNLIKELY(condition) (__builtin_expect(!!(condition), 0))
#else
#define mindfour_UNLIKELY(condition) (!!(condition))
#endif
#endif

#ifndef mindfour_UNPREDICTABLE
#if defined(__has_builtin)
#if __has_builtin(__builtin_unpredictable)
#define mindfour_UNPREDICTABLE(condition) (__builtin_unpredictable(!!(condition)))
#endif
#endif
#endif
#ifndef mindfour_UNPREDICTABLE
#if defined(__GNUC__) && (__GNUC__ >= 9) && !defined(__INTEL_COMPILER)
#define mindfour_UNPREDICTABLE(condition) (__builtin_expect_with_probability(!!(condition), 0, 0.5))
#else
#define mindfour_UNPREDICTABLE(condition) (!!(condition))
#endif
#endif

// Disable padding for structure members.
#ifndef mindfour_DENSELY_PACKED_STRUCTURE
#if defined(__GNUC__)
#define mindfour_DENSELY_PACKED_STRUCTURE __attribute__((__packed__))
#else
#error "Compiler-specific implementation of mindfour_DENSELY_PACKED_STRUCTURE required"
#endif
#endif

#ifndef mindfour_ALIGN
#if defined(__GNUC__)
#define mindfour_ALIGN(alignment) __attribute__((__aligned__(alignment)))
#elif defined(_MSC_VER)
#define mindfour_ALIGN(alignment) __declspec(align(alignment))
#else
#error "Compiler-specific implementation of mindfour_ALIGN required"
#endif
#endif

#ifndef mindfour_FORCE_INLINE
#if defined(__GNUC__)
#define mindfour_FORCE_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define mindfour_FORCE_INLINE __forceinline
#else
#define mindfour_FORCE_INLINE inline
#endif
#endif

/***** Symbol visibility macros *****/

#ifndef mindfour_INTERNAL_SYMBOL
#if defined(__ELF__)
#define mindfour_INTERNAL_SYMBOL __attribute__((__visibility__("internal")))
#elif defined(__MACH__)
#define mindfour_INTERNAL_SYMBOL __attribute__((__visibility__("hidden")))
#else
#define mindfour_INTERNAL_SYMBOL
#endif
#endif
