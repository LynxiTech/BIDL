# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

# This code references the source code of OpenMMLab projects, which are
# licensed under the Apache License, Version 2.0.
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_info_from_json(json_file, col_names=('epoch', 'accuracy_top-1', 'accuracy_top-5'), mode='val'):
    assert mode == 'val'
    with open(json_file, 'r') as f:
        lines = f.readlines()
    json_str = '{"infos": [' + ','.join(lines[1:]) + ']}'
    infos_list = json.loads(json_str)['infos']
    infos_pd = pd.DataFrame(infos_list)
    col_mode = infos_pd.get('mode').values
    cols_list = [infos_pd.get(_).values for _ in col_names]
    items_filt = [__ for _, __ in zip(col_mode, list(zip(*cols_list))) if _ == mode]
    cols_dict = {_: __ for _, __ in zip(col_names, list(zip(*items_filt)))}
    return cols_dict


def main():
    col_names = ('epoch', 'accuracy_top-1', 'accuracy_top-5')
    # col_names = ('epoch', 'mAP', 'AP50')
    ncol = len(col_names) - 1

    base_fold = '../work_dirs/'
    dns = [
        'alex3c2flif-b16x1-dvsgesture'
    ]

    # base_fold = '../../!!!!!si!!!!!/mmdet.work_dirs.new/'
    # dns = [
    #     'fcos_hr18-si12122200-imagenet-voc',
    #     'fcos_hr18-std-imagenet-voc',
    # ]

    fig, axes = plt.subplots(1, ncol)
    _start1 = 1  # 31  # XXX 31  1
    _start2 = 1  # -30  # XXX 61  9

    for dn in dns:
        # find json log file, read log by json
        sub_fold = os.path.join(base_fold, dn)
        json_files = glob.glob(sub_fold + '/*.log.json')
        assert len(json_files) == 1
        json_file = json_files[0]
        print(json_file)
        cols_dict = read_info_from_json(json_file, col_names)
        # viz by items
        for i in range(ncol):
            linestyle = '-'
            xs, ys = cols_dict[col_names[0]], cols_dict[col_names[i + 1]]
            xs, ys = xs[_start1:], ys[_start1:]
            _mean = np.mean(ys[_start2:])
            _std = np.std(ys[_start2:])
            axes[i].plot(xs, ys, label=f'{dn[5:]} {_mean:.4f} {_std:.4f}', linestyle=linestyle)

    [_.legend() for _ in axes]
    # fig.legend(loc='upper left')
    # axes[0].legend(loc='lower left')
    plt.show()
    # fig.show(); fig.waitforbuttonpress()


if __name__ == '__main__':
    main()
