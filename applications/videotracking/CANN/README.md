CANN: Implementing Tiger1 Dataset Object Tracking Using Continuous Attractor Subnetworks

## Dataset Preparation
First, download the Tiger1 dataset to the "./data/" directory.

## Environment Setup
python3.8, pytorch, lyngor 

## Script Introduction
```Bash
cal.py encapsulates functions for generating Success plot graphs and calculating AUC.
cann.py is the script for constructing the Continuous Attractor Model.
compile.py is the compilation script.
infer_apu.py and infer_gpu.py are the inference scripts for running on the corresponding hardware.
Note that before running infer_apu.py, you need to run compile.py to produce the compiled artifacts.
```

## GPU Inference
```Bash
python infer_gpu.py
```

## APU Inference
```Bash
python compile.py
python infer_apu.py
```

