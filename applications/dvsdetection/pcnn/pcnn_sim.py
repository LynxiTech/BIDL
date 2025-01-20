import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math
import os
import glob
import matplotlib.pyplot as plt
import argparse
import time
import re
import sys; sys.path.append('../../../')

class SideSuppressFilter(nn.Module):
    def __init__(self):
        super(SideSuppressFilter, self).__init__()
        W = torch.tensor([[[[0.025, 0.025, 0.025, 0.025, 0.025],
                               [0.025, 0.075, 0.075, 0.075, 0.025],
                               [0.025, 0.075, 0,     0.075, 0.025],
                               [0.025, 0.075, 0.075, 0.075, 0.025],
                               [0.025, 0.025, 0.025, 0.025, 0.025]]]], dtype=torch.float32)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv2d.weight.detach().copy_(W)

    def forward(self, x):
        C = self.conv2d(x)
        F = x - C
        F = F * 5
        return F

class MotionDetectionModel(nn.Module):
    def __init__(self):
        super(MotionDetectionModel, self).__init__()

        self.side_suppress_filter = SideSuppressFilter()

        W_1 = torch.tensor([[[[1, 0, 1],
                            [0, 0, 0],
                            [1, 0, 1]]]], dtype=torch.float32)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2d_1.weight.detach().copy_(W_1)

        W_2 = torch.tensor([[[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]]], dtype=torch.float32)
        self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv2d_2.weight.detach().copy_(W_2)

    def forward(self, x):
        F = self.side_suppress_filter(x)
        T = 180.0 * torch.ones_like(x)
        O = torch.zeros_like(x)

        for i in range(0, 5):
            C1 = self.conv2d_1(O)
            C2 = self.conv2d_2(O)
            Fh = F + 30 * C1 + 30 * C2
            X = (Fh >= T).float()
            T = X * 180.0 * torch.ones_like(T) + (1 - X) * T
            O = O + X

        O = (O >= 1).float()

        return O

parser = argparse.ArgumentParser()
parser.add_argument('--compile_apu', default=0, type=int, help='0:disable, 1:enable')
parser.add_argument('--device', default='cpu:0', type=str, help='cpu:0/apu:0/gpu:0' )
parser.add_argument('--render', default=0, type=int, help='0:disable, 1:enable')
args = parser.parse_args()


root_path = '../../../'
img_path = root_path + "data/data_pcnn/"
generated_path = root_path + "model_files/dvsdetection/pcnn/"
exit_flag = False
run_apu = 0

if __name__ == '__main__':
    if re.search(r'(cuda:(\d+)|cpu:(\d+))', args.device):
        device = args.device
    elif re.search(r'apu:(\d+)', args.device):
        device = int(re.search(r'apu:(\d+)', args.device).group(1))
        run_apu = 1
        
    if args.compile_apu == 1:
        model = MotionDetectionModel()
        from lynadapter.lyn_compile import model_compile
        model_compile(model,  _base_= generated_path, in_size=[[[1,1,256,256]]], version=1, batch_size=1, input_type='float32')
        
    if run_apu == 1:
        from lynadapter.lyn_sdk_model import ApuRun_Single
        model = ApuRun_Single(apu_device=device, apu_model_path=generated_path+'Net_0')
    else:
        model = MotionDetectionModel().to(device)
        model.eval()

    data_path = os.path.join(img_path, '*.bmp')  
    img_data = glob.glob(data_path)
    def sort_by_number(file_path):
        number = int(''.join(filter(str.isdigit, file_path)))
        return number
    img_data = sorted(img_data, key=sort_by_number)

    if args.render == 1:
        def on_key_press(event):
            global exit_flag
            if event.key == 'q':  
                exit_flag = True
                plt.close()
        plt.connect('key_press_event', on_key_press)

    frames = []
    total_time = 0
    img_num = 0
    for img_path in img_data:  
        img_pack = Image.open(img_path).convert('L')
        img_pack = np.array(img_pack)
        image1 = img_pack
        img_pack = np.expand_dims(img_pack, axis=0)
        img_pack = np.expand_dims(img_pack, axis=0)
        start = time.time()
        if run_apu == 1:
            out = model.run(img_pack.astype(np.float32), 1)
            out = torch.from_numpy(out[0])
        else:
            with torch.no_grad():
                img_pack = torch.tensor(img_pack, dtype=torch.float32).to(device)
                out = model(img_pack).cpu().numpy()
        end = time.time()
        total_time += end - start
        img_num += 1
        
        image2 = out.squeeze()[5:-5, 5:-5]
        image1 = image1[5:-5, 5:-5]

        if args.render == 1:
            plt.subplot(1, 2, 1)
            plt.imshow(image1, cmap='gray')
            plt.title('Image 1')

            plt.subplot(1, 2, 2)  
            plt.imshow(image2, cmap='gray')
            plt.title('Image 2')

            plt.draw()
            plt.pause(0.001)
            plt.clf()

            if exit_flag:
                break
    plt.close()
    if run_apu == 1:
        model.apu_unload()
    print('fps = {:.2f}, img_num = {}, total_time = {:.5f}'.format(img_num/total_time, img_num, total_time))

        