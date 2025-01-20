DVS High-speed Turntable Dataset Object Detection: Based on Lingxi Technology's proprietary DVS High-speed Turntable Dataset and Styolo network, achieve multi-class object detection in a rapidly rotating turntable.

## Support Model Training and Inference on GPU
GPU runtime environment is consistent with the styolo task
To continue training with --resume, get the ckpt from https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt

## Support Model Inference on APU
APU runtime environment is consistent with the styolo task

## Dataset
For this task, you need to use the high-speed turntable dataset "dvs_turntable". Please download the dataset from the dataset resource package.

## Pre-trained Checkpoints
Please download the pre-trained weights from the weight resource package.

### GPU Training  
```Bash
python train.py 
By default, it uses the "dvs_turntable" dataset to train the styolov5n model.
You can specify the model using --cfg and the dataset using --data.
```
### GPU Inference 
```Bash
python val.py 
By default, it uses the "dvs_turntable" val dataset and performs inference using the styolov5n model.
You can specify the weights using --weights and the dataset using --data.
```
### GPU real-time detection
```Bash
python detect.py 
By default, it uses the "dvs_turntable" test dataset and performs inference using the styolov5n model.
You can specify the weights using --weights and the dataset using --data.
```


### APU Inference 
```Bash
python apu_infer.py --c 1
By default, it uses the "dvs_turntable" val dataset and performs inference using the styolov5n model.
You can specify the weights using --weights and the dataset using --data.
The --c parameter can be used to specify whether to recompile the model (YES/NO: 1/0).
```
### APU real-time detection
```Bash
python apu_detect.py --c 1
By default, it uses the "dvs_turntable" test dataset and performs inference using the styolov5n model.
You can specify the weights using --weights and the dataset using --data.
The --c parameter can be used to specify whether to recompile the model (YES/NO: 1/0).
```
