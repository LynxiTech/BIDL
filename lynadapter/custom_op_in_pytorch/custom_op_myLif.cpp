/*
    custom op in torch
*/

#include <torch/script.h>
#include "custom_op_myLif.h"


torch::Tensor reset_with_decay(torch::Tensor input,double theta,double v_0,double alpha,double beta) {
  torch::Tensor output;
  output = input;
  return output;
}

torch::Tensor cmp_and_fire(torch::Tensor input,double theta) {
  torch::Tensor output;
  output = input;
  return output;
}





/* op load definition*/
torch::Tensor load(torch::Tensor input, string flag, bool is_initial, std::vector<int64_t> shape,
                   string dtype, int64_t offset, int64_t step, bool uselookup, int64_t mode, int64_t group_id) {
  return input.clone();
}

/* op save definition*/
torch::Tensor save(torch::Tensor input, string flag, int64_t offset, int64_t step, bool isOutput, bool uselookup, int64_t mode, int64_t group_id) {
  return input.clone();
}


/* register operators load */
static auto registry_5 =
    torch::RegisterOperators()
        .op("custom::load", &load)
        .op("custom::save", &save);

static auto registry_6 =
    torch::RegisterOperators()
        // We parse the schema for the user.
        .op("custom::resetwithdecay", &reset_with_decay)
        .op("custom::cmpandfire", &cmp_and_fire);

