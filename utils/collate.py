# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.

from torch.utils.data.dataloader import default_collate  
from collections.abc import Sequence, Mapping  
  
def collate(batch, samples_per_gpu=1):  

    if isinstance(batch[0], str):  

        return batch[0]
    
    if isinstance(batch[0], Sequence):  
        transposed = list(zip(*batch))  
        result = []  
        for samples in transposed:  
            result.append(collate(samples, samples_per_gpu))  

        return result  
      
    elif isinstance(batch[0], Mapping):  
        result = {}              
        for key in batch[0]:  
            values = [d[key] for d in batch]  
            result[key] = collate(values, samples_per_gpu)  
           
        return result  
      
    else:  
        return default_collate(batch)
  