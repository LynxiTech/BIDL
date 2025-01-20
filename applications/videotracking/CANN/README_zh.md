CANN: 使用连续吸引子网络，实现Tiger1数据集目标追踪

## 数据集准备
首先下载Tiger1数据集至./data/目录下

## 环境准备
python3.8, pytorch, lyngor 

## 脚本介绍
```Bash
cal.py中封装了一些生成Success plot图和计算AUC所需的函数。
cann.py为连续吸引子模型构建脚本。
compile.py为编译脚本。
infer_apu.py和infer_gpu.py分别为在对应硬件上的推理脚本。
注意，运行infer_apu.py前需要先运行compile.py生产编译生成物。
```

## GPU推理
```Bash
python infer_gpu.py
```

## APU推理
```Bash
python compile.py
python infer_apu.py
```

