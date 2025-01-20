ST-YOLO: Event Stream Data Object Detection based on Spatio-Temporal Dynamics.

### First, compile the dynamic library
- Execute ./build_run_lif.sh to compile and generate the dynamic library.

## Training and Inference on GPU
- To run the model training and inference on a GPU, install python3.9 first, then run:
```Bash
pip install -r requirements_gpu.txt
```

## Inference on APU
- To run the model inference on an APU, install python3.8 first.
- Consistent with bidl home directory environment.

## Required Data
To evaluate or train ST-YOLO, please download the required preprocessed dataset.
Use the following command to download the dataset on Linux:
```Bash
wget https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar
```

For dataset preprocessing, refer to the link[link](scripts/genx/README.md).

Modify the dataset path and related configurations in config/dataset/gen1.yaml.

## Pre-trained Checkpoints
### Gen1
```Bash
st_yolov5n: epoch=001-step=394428-val_AP=0.41.ckpt
st_yolov5s: epoch=001-step=394428-val_AP=0.44.ckpt
st_yolov5x: epoch=000-step=197214-val_AP=0.46.ckpt
```
### Training on GPU
```Bash
python train.py 
You will be prompted to register wandb (tracking and visualization tool) on the first run. Follow the prompts to register/login.
You can modify the configurations in config/train.yaml and its sub-files.
train.yaml will refer to general.yaml (training-related parameter settings), dataset/gen1.yaml (dataset settings, modify dataset path), model/yolodet.yaml (model settings, modify yolo_dict for the desired model).
```
### Inference on GPU
```Bash
python gpu_infer.py 
You can modify the configurations in config/val_gpu.yaml. val_gpu.yaml refers to dataset/gen1.yaml and model/yolodet.yaml.
If you need to modify dataset paths or other settings, modify dataset/gen1.yaml. To change the model configuration, modify yolo_dict in model/yolodet.yaml and checkpoint in val_gpu.yaml accordingly.
```
### Real-time Demonstration on GPU 
```Bash
python gpu_show.py 
You can modify the configurations in config/gpu_show.yaml. gpu_show.yaml refers to dataset/demo.yaml and model/yolodet.yaml.
If you need to modify dataset paths or other settings, modify dataset/demo.yaml. To change the model configuration, modify yolo_dict in model/yolodet.yaml and checkpoint in gpu_show.yaml accordingly.

Real-time inference results are saved in the visualize/pred/ directory. On Linux, you can use ffmpeg in this directory to convert them to video format, for example:
ffmpeg -framerate 20 -i %04d.png -c:v libx264 output_name.mp4
```

### APU Compilation
```Bash
python compile_backbone.py --model yolov5s   Compile model backbone
python compile_head.py --model yolov5s  Compile model head
python compile_styolo.py --model yolov5s   Compile the entire model
The --model parameter can be set as yolov5n, yolov5s, or yolov5x.
```
### APU Inference
```Bash
python apu_infer.py --model yolov5s --c 0
By default, inference is performed using the yolov5s model. You can specify the model as yolov5n, yolov5s, or yolov5x using the --model parameter. The --c parameter specifies whether to recompile the model (YES/NO: 1/0).

To modify the dataset path for demonstration, refer to config/dataset/gen1.yaml.
```
### Real-time Demonstration on APU
```Bash
python apu_show.py --model yolov5s --c 0
By default, inference is performed using the yolov5s model. You can specify the model as yolov5n, yolov5s, or yolov5x using the --model parameter. The --c parameter specifies whether to recompile the model (YES/NO: 1/0).

To modify the dataset path for demonstration, refer to config/dataset/demo.yaml.
Real-time inference results are saved in the visualize/pred/ directory. Conversion to video format can be done as mentioned before.
```

