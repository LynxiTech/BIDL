/*
    custom op in torch
*/

#include <torch/script.h>
#include "custom_op_Rand.h"

/* op Rand definition*/
torch::Tensor Rand(
    torch::Tensor tensor,
    c10::ArrayRef<int64_t> o_shape,
    int64_t uInitValue,
    int64_t mode) {
  torch::Tensor output = torch::rand(o_shape);
  return output;
}

/* register operators Rand */
static auto registry =
    torch::RegisterOperators()
        // We parse the schema for the user.
        .op("custom::Rand", &Rand);