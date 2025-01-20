# pcnn
基于PCNN(脉冲耦合神经网络)和侧抑制网络对输入图像进行滤波，实现背景抑制和目标增强 。用于红外运动弱小目标的检测任务。

## 编译 
python3 pcnn_sim.py --compile_apu 1 --device apu:0  --render 1

## 只推理

### cpu
python3 pcnn_sim.py --compile_apu 0 --device cpu:0  --render 1

### gpu
python3 pcnn_sim.py --compile_apu 0 --device cuda:0  --render 1

### apu
python3 pcnn_sim.py --compile_apu 0 --device apu:0  --render 1