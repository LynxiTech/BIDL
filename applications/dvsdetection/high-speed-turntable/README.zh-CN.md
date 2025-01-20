DVS高速转盘数据集目标检测：基于灵汐科技自有的DVS高速转盘数据集及styolo网络，实现高速转动的转盘中的多类目标检测。

## GPU上可跑模型训练和推理
GPU运行环境和styolo任务一致。
如需使用--resume继续训练，请从https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt下载权重

## APU上可跑模型推理
APU运行环境和styolo任务一致

## 数据集
本任务需要使用高速转盘数据集dvs_turntable，数据集请从数据集资源包下载。

## 预训练权重
本任务预训练权重请从权重资源包下载。

### GPU训练  
```Bash
python train.py 
默认使用dvs_turntable数据集，训练styolov5n模型。
可通过--cfg指定模型，--data指定数据集。
```
### GPU推理  
```Bash
python val.py 
默认使用dvs_turntable验证集，styolov5n模型进行推理。
可通过--weights指定模型，--data指定数据集。
```
### GPU实时检测
```Bash
python detect.py 
默认使用dvs_turntable测试集，styolov5n模型进行推理。
可通过--weights指定模型，--data指定数据集。
```


### APU推理  
```Bash
python apu_infer.py --c 1
默认使用dvs_turntable验证集，styolov5n模型进行推理。
可通过--weights指定模型，--data指定数据集。
--c参数可指定是否重新编译模型(YES/NO: 1/0)
```
### APU实时检测  
```Bash
python apu_detect.py --c 1
默认使用dvs_turntable测试集，styolov5n模型进行推理。
可通过--weights指定模型，--data指定数据集。
--c参数可指定是否重新编译模型(YES/NO: 1/0)
```

