#pragma once

#ifdef _WIN32
#if defined(torchcluster_EXPORTS)
#define CLUSTER_API __declspec(dllexport)
#else
#define CLUSTER_API __declspec(dllimport)
#endif
#else
#define CLUSTER_API
#endif

#if (defined __cpp_inline_variables) || __cplusplus >= 201703L
#define CLUSTER_INLINE_VARIABLE inline
#else
#ifdef _MSC_VER
#define CLUSTER_INLINE_VARIABLE __declspec(selectany)
#else
#define CLUSTER_INLINE_VARIABLE __attribute__((weak))
#endif
#endif
