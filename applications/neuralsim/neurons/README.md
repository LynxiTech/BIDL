## Introduction

The neuralsim folder provides a detailed introduction to the neural simulation tool.

## Quick Start


### To run existing neural models, execute the following commands in the applications/neuralsim/neuron directory:
  ```Python
python3 test.py --neuron lif --use_lyngor 1  --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron adex --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron izhikevich --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron multicompartment --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron hh --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
python3 test.py --neuron multicluster --use_lyngor 1 --use_legacy 0 --use_gpu 1 --plot 1 --device 0
  ``` 
### Running stdp demo
  ```Python
python3 test_stdp.py --use_lyngor 1  --use_legacy 0 --use_gpu 1 --plot 1 --device 0
  ```