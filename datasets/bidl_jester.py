# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import os
import shutil

import cv2
import numpy as np

from .base_dataset import BaseDataset

class Jester20bn(BaseDataset):
    """
    Suggested by https://blog.csdn.net/weixin_43838639/article/details/109454375.
    Labels downloaded from https://www.kaggle.com/datasets/kylecloud/jester-csv,
    Videos downloaded from
        https://www.kaggle.com/datasets/zhaochengdu1998/jester1
        https://www.kaggle.com/datasets/zhaochengdu1998/jester2
    Note: videos from https://www.kaggle.com/datasets/kylecloud/20bn-jester-v1-videos are not integral!
    """
    CLASSES = [
        'Doing other things',
        'Drumming Fingers',
        'No gesture',
        'Pulling Hand In',
        'Pulling Two Fingers In',
        'Pushing Hand Away',
        'Pushing Two Fingers Away',
        'Rolling Hand Backward',
        'Rolling Hand Forward',
        'Shaking Hand',
        'Sliding Two Fingers Down',
        'Sliding Two Fingers Left',
        'Sliding Two Fingers Right',
        'Sliding Two Fingers Up',
        'Stop Sign',
        'Swiping Down',
        'Swiping Left',
        'Swiping Right',
        'Swiping Up',
        'Thumb Down',
        'Thumb Up',
        'Turning Hand Clockwise',
        'Turning Hand Counterclockwise',
        'Zooming In With Full Hand',
        'Zooming In With Two Fingers',
        'Zooming Out With Full Hand',
        'Zooming Out With Two Fingers'
    ]

    def load_annotations(self):
        anno_file = os.path.join(self.data_prefix, self.ann_file)
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        data_infos = []
        for line in lines[1:]:
            video, clssa, label, frames, height, width = line.strip().split(',')
            video = os.path.join(self.data_prefix, 'frames', video)
            label = int(label)
            frames = frames.split('|')
            if len(frames) < 16:
                continue  # import pdb; pdb.set_trace()
            shape = [int(height), int(width)]
            info = {
                'video': video, 'frames': frames, 'shape': shape,
                'gt_label': np.array(label, dtype='int64'),
            }
            data_infos.append(info)
        return data_infos

    @staticmethod
    def convert_csv(vid_prefix, csv_i, csv_o, mapping):
        """For organizing dataset."""
        with open(csv_i, 'r') as fi:
            lines_i = fi.readlines()

        lines_o = [f'video,class,label,frames,height,width\n']
        for i, line_i in enumerate(lines_i):
            if i % 1000 == 0:
                print(i, line_i)

            video_class = line_i.strip().split(';')
            if len(video_class) == 2:
                video, clssa = video_class
                label = mapping[clssa]
            elif len(video_class) == 1:
                video, clssa = video_class[0], ''
                label = '-1'
            else:
                raise NotImplementedError

            video_path = vid_prefix + video
            fns = os.listdir(video_path)
            fns.sort()
            frames = '|'.join(fns)

            img_path = video_path + '/' + fns[0]
            height, width = cv2.imread(img_path).shape[:2]

            line_o = f'{video},{clssa},{label},{frames},{height},{width}\n'
            lines_o.append(line_o)

        with open(csv_o, 'w') as fo:
            fo.writelines(lines_o)

    def wrap_video(self, video_file):
        """For Flask demo."""
        video_file = os.path.abspath(video_file)
        frame_fold = os.path.join('.tmp/', video_file.split('/')[-1])
        if not os.path.exists(frame_fold):
            os.makedirs(frame_fold)

        print(f'extract input video file to local ``{frame_fold}``')
        cnt = 0
        frame_fns = []
        h, w = -1, -1
        vcap = cv2.VideoCapture(video_file)
        while True:
            flag, frame = vcap.read()
            if cnt == 0:
                h, w = frame.shape[:2]
            if flag is False:
                vcap.release()
                break
            frame_fn = f'{cnt:06d}.jpg'
            cv2.imwrite(os.path.join(frame_fold, frame_fn), frame)
            frame_fns.append(frame_fn)
            cnt += 1

        assert len(frame_fns) > 0
        info = {
            'video': frame_fold, 'frames': frame_fns, 'shape': [h, w],
            'gt_label': np.array(-1, dtype='int64')
        }
        xmpl = self.pipeline(info)

        print(f'delete input video frames in local ``{frame_fold}``')
        shutil.rmtree(frame_fold)

        return xmpl


########################################################################################################################


def main():
    """
    Firstly,
      download label files from https://www.kaggle.com/datasets/kylecloud/jester-csv
      download frame files from
          https://www.kaggle.com/datasets/zhaochengdu1998/jester1
          and
          https://www.kaggle.com/datasets/zhaochengdu1998/jester2
    Secondly,
      extract label files to your dataset directory ``20bn-jester-v1/labels/``
      extract frame files both ``jester1.zip`` and ``jester2.zip`` to ``20bn-jester-v1/frames/``
    Lastly,
      execute the following code to generate labels ready for training.
    """
    base_fold = '/media/lynxitech/Storage/Active/Datasets/20bn-jester-v1/'
    csv_prefix = base_fold + 'labels/'
    vid_prefix = base_fold + 'frames/'

    csv_fns = [
        ['jester-v1-train.csv', 'train.csv'],
        ['jester-v1-validation.csv', 'val.csv'],
        ['jester-v1-test.csv', 'test.csv']
    ]
    mapping = dict(zip(Jester20bn.CLASSES, range(len(Jester20bn.CLASSES))))

    for csv_fn_i, csv_fn_o in csv_fns:
        csv_i = csv_prefix + csv_fn_i
        csv_o = base_fold + csv_fn_o
        print(f'Convert from {csv_fn_i} to {csv_fn_o}...')
        Jester20bn.convert_csv(vid_prefix, csv_i, csv_o, mapping)


if __name__ == '__main__':
    main()
