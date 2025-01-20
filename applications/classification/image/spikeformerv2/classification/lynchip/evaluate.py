# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.

import sys; 
sys.path.append('../')
sys.path.append('../../../../../../')
import torch
from timm.utils import accuracy
import util.misc as misc
from utils import globals

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    isApu = globals.get_value('isApu')
    if isApu:
        generated_path = globals.get_value('generated_path')
        apu_id = globals.get_value('apu_id')
        from lynadapter.lyn_sdk_model import ApuRun_Single
        model = ApuRun_Single(apu_device=apu_id, apu_model_path=generated_path+'/Net_0')

    batch_id = 0
    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            if isApu:
                output_apu = model(images.numpy())
                output_apu = torch.from_numpy(output_apu[0]).to(torch.float32)
                output = output_apu
            loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        batch_id += 1

    if isApu:
        model.apu_unload()
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


