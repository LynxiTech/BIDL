ST-YOLO: 基于时间空间动力学，进行事件流数据目标检测。

### 首先编译动态库
- 执行``./build_run_lif.sh``，即可编译生成动态库

## GPU上可跑模型训练和推理
- GPU环境安装，需要python3.9，随后运行：
```Bash
pip install -r requirements_gpu.txt
```

## APU上可跑模型推理
- APU环境安装，需要python3.8
- 与bidl主目录环境一致

## Required Data
评估或训练 ST-YOLO，请下载所需的预处理数据集。
Linux可使用以下指令下载：
```Bash
wget https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar
```

数据集预处理请参考[链接](scripts/genx/README.md).

数据集使用路径及相关配置请修改config/dataset/gen1.yaml

## Pre-trained Checkpoints
### Gen1
```Bash
st_yolov5n: epoch=001-step=394428-val_AP=0.41.ckpt
st_yolov5s: epoch=001-step=394428-val_AP=0.44.ckpt
st_yolov5x: epoch=000-step=197214-val_AP=0.46.ckpt
```
### GPU训练  
```Bash
python train.py 
第一次运行会提示需要注册wandb（跟踪、可视化工具），请按提示注册登录。
相关配置可修改config/train.yaml及其下属配置文件。
train.yaml会调用general.yaml（训练相关参数设置），dataset/gen1.yaml(数据集设置，可修改数据集路径)，model/yolodet.yaml(模型设置，修改其中的yolo_dict为所需要的模型)
```
### GPU推理  
```Bash
python gpu_infer.py 
相关配置可修改config/val_gpu.yaml。val_gpu.yaml会调用dataset/gen1.yaml和model/yolodet.yaml。
如需修改数据集路径等设置，可修改dataset/gen1.yaml。修改模型配置请对应修改model/yolodet.yaml中的yolo_dict和val_gpu.yaml中的checkpoint。
```
### GPU实时演示  
```Bash
python gpu_show.py 
相关配置可修改config/gpu_show.yaml。gpu_show.yaml会调用dataset/demo.yaml和model/yolodet.yaml。
如需修改数据集路径等设置，可修改dataset/demo.yaml。修改模型配置请对应修改model/yolodet.yaml中的yolo_dict和gpu_show.yaml中的checkpoint。

实时推理结果保存在visualize/pred/目录下，Linux系统可在此目录下使用ffmpeg将其转为视频格式，例如：
ffmpeg -framerate 20 -i %04d.png -c:v libx264 output_name.mp4
```

### APU编译 
```Bash
python compile_backbone.py --model yolov5s   模型backbone编译
python compile_head.py --model yolov5s  模型head编译
python compile_styolo.py --model yolov5s   整个模型编译
其中--model参数可指定模型为yolov5n，yolov5s或yolov5x
```
### APU推理  
```Bash
python apu_infer.py --model yolov5s --c 0
默认使用yolov5s模型进行推理，可使用--model参数指定模型为yolov5n，yolov5s或yolov5x, --c参数可指定是否重新编译模型(YES/NO: 1/0)

演示使用数据集路径修改见 config/dataset/gen1.yaml
```
### APU实时演示
```Bash
python apu_show.py --model yolov5s --c 0
默认使用yolov5s模型进行推理，可使用--model参数指定模型为yolov5n，yolov5s或yolov5x, --c参数可指定是否重新编译模型(YES/NO: 1/0)

演示使用数据集路径修改见 config/dataset/demo.yaml
实时推理结果保存在visualize/pred/目录下，转视频格式操作同上。
```

