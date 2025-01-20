## 简介

neuralsim文件夹是神经元仿真工具的详细介绍

## 快速使用



### 在applications/neuralsim/neuron目录下执行如下命令，即可运行已有的神经元模型
  ```Python
python3 test.py --neuron lif --use_lyngor 1  --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron adex --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron izhikevich --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron multicompartment --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron hh --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron multicluster --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
  ``` 
### stdp demo运行
  ```Python
python3 test_stdp.py --use_lyngor 1  --use_legacy 0 --use_gpu 1 --plot 1 --device 0
  ```