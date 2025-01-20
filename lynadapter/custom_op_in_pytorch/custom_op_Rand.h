/*
    custom op in torch
*/

#include <torch/script.h>
#include <cstddef>
#include <vector>
#include <string>

// clang-format off
#  if defined(_WIN32)
#    if defined(custom_ops_EXPORTS)
#      define CUSTOM_OP_API __declspec(dllexport)
#    else
#      define CUSTOM_OP_API __declspec(dllimport)
#    endif
#  else
#    define CUSTOM_OP_API
#  endif
// clang-format on

/* op Rand interface declare*/
CUSTOM_OP_API torch::Tensor Rand(
    torch::Tensor tensor,
    c10::ArrayRef<int64_t> o_shape,
    int64_t uInitValue,
    int64_t mode = 0);