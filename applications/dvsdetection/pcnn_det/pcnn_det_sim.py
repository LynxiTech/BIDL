from matplotlib import axis
import torch
import torch.nn as nn
import torch.nn.functional as torchfunc
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import math
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import argparse
import time
import re
import cv2
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
        self.scale_factor = 100.0

    def forward(self, x):
        C = self.conv2d(x)
        F = x * self.scale_factor - C * self.scale_factor
        F = F * 3.0 / self.scale_factor
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
        T = 210.0 * torch.ones_like(x)
        O = torch.zeros_like(x)
        for i in range(0, 5):
            C1 = self.conv2d_1(O)
            C2 = self.conv2d_2(O)
            Fh = F + 30 * C1 +  30 * C2
            X = (Fh >= T).float()
            T = X * 220.0 * torch.ones_like(T) + (1 - X) * (T-20)
            O = O + X

        O = (O >= 1).float()

        return O

def preprocess(image_path):
    # Open the image
    image = cv2.imread(image_path)

    # Resize the image
    resized_image = cv2.resize(image, (256, 256))

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Invert the image
    inverted_image = cv2.bitwise_not(grayscale_image)

    # Enhance contrast
    enhanced_image = cv2.convertScaleAbs(inverted_image, alpha=2.0, beta=-100)

    return image, enhanced_image

def find_bounding_box(image):
    image = image.astype(np.uint8)

    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Get the largest contour (assuming it corresponds to the brightest object)
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 0:
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center coordinates of the bounding rectangle
            center_x = x + w // 2
            center_y = y + h // 2

            # Calculate the start and end coordinates of the bounding rectangle
            start_x = center_x - w // 2
            start_y = center_y - h // 2
            end_x = start_x + w
            end_y = start_y + h

            return [start_x, start_y, end_x, end_y]

    return []


parser = argparse.ArgumentParser()
parser.add_argument('--compile_apu', default=0, type=int, help='0:disable, 1:enable')
parser.add_argument('--device', default='cpu:0', type=str, help='cpu:0/apu:0/gpu:0' )
parser.add_argument('--render', default=0, type=int, help='0:disable, 1:enable')
args = parser.parse_args()

root_path = '../../../'
img_path = root_path + "data/data_pcnn_det/"
generated_path = root_path + "model_files/dvsdetection/pcnn_det/"
run_apu = 0
video_w = 720
video_h = 720

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
    
    total_time = 0
    img_num = 0
    
    video_output = './pcnn_det_video.mp4'
    fps = 10.0  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output, fourcc, fps, (video_w, video_h), isColor=True)

    for index, item in enumerate(img_data):
        print('frame ', index)
        img_ori, img_pre = preprocess(item)
        img_pre = np.expand_dims(img_pre, axis=0)
        img_pre = np.expand_dims(img_pre, axis=0)
        start = time.time()
        if run_apu == 1:
            out = model.run(img_pre.astype(np.float32), 1)
            out = out[0]
        else:
            with torch.no_grad():
                img_pre = torch.tensor(img_pre, dtype=torch.float32).to(device)
                out = model(img_pre).cpu().numpy()
        end = time.time()
        total_time += end - start
        img_num += 1
        
        out = out.squeeze()
        out[:5, :] = 0   # Top boundary
        out[-5:, :] = 0  # Bottom boundary
        out[:, :5] = 0   # Left boundary
        out[:, -5:] = 0  # Right boundary

        bbox = find_bounding_box(out)
        if len(bbox) > 0:
            start_x, start_y, end_x, end_y = bbox
            scale_x = img_ori.shape[1] / 256
            scale_y = img_ori.shape[0] / 256
            boxed_img = cv2.rectangle(img_ori, (int(start_x*scale_x), int(start_y*scale_y)), (int(end_x*scale_x), int(end_y*scale_y)), (0,255,0), 2)
            frame = boxed_img
            text = f"({start_x}, {start_y}) - ({end_x}, {end_y})"
            cv2.putText(frame, text, (int(start_x*scale_x), int(start_y*scale_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            frame = img_ori

        frame = cv2.resize(frame, (video_w, video_h))
        video_writer.write(frame)

        if args.render == 1:
            cv2.imshow('pcnn_det',frame)
            wait = cv2.waitKey(100)
            
    video_writer.release()
    if args.render == 1:
        cv2.destroyAllWindows()
    
    if run_apu == 1:
        model.apu_unload()
    print('fps = {:.2f}, img_num = {}, total_time = {:.5f}s'.format(img_num/total_time, img_num, total_time))

        
