# pcnn
PCNN (Pulse Coupled Neural Network) is used to filter input images, achieving background suppression and target enhancement. It is employed for the detection of weak infrared moving targets.

## Compile 
python3 pcnn_det_sim.py --compile_apu 1 --device apu:0  --render 1

## Inference Only 

### cpu
python3 pcnn_det_sim.py --compile_apu 0 --device cpu:0  --render 1

### gpu
python3 pcnn_det_sim.py --compile_apu 0 --device cuda:0  --render 1

### apu
python3 pcnn_det_sim.py --compile_apu 0 --device apu:0  --render 1