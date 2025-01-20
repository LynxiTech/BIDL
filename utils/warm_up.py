# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

def warm_up(cur_iters,warmup_iters,warmup_ratio,lr):
    if cur_iters<=warmup_iters:
        k = (1 - cur_iters / warmup_iters) * (1 -warmup_ratio)
        warmup_lr = lr * (1 - k) 
    else:
        warmup_lr=lr
    return warmup_lr