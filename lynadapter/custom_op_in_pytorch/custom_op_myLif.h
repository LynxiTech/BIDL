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
using namespace std;
/* op my_lif interface declare*/

/* op load interface declare*/
CUSTOM_OP_API torch::Tensor load(torch::Tensor tensor, string flag, bool is_initial,
                                 std::vector<int64_t> shape, string dtype, int64_t offset,
                                 int64_t step,bool uselookup, int64_t mode, int64_t group_id);

/* op save interface declare*/
CUSTOM_OP_API torch::Tensor save(torch::Tensor tensor, string flag, int64_t offset,
                                 int64_t step, bool isOutput,bool uselookup, int64_t mode, int64_t group_id);

CUSTOM_OP_API torch::Tensor reset_with_decay(torch::Tensor tensor,double theta,double v_0,double alpha,double beta);
CUSTOM_OP_API torch::Tensor cmp_and_fire(torch::Tensor tensor,double theta);