## Execution Instructions
### 1. Compile and Run Existing Models
Note: The default storage path for the compiled artifacts of all models is under the project root directory in ./model_files, specifically in ./model_files/app category/compiled artifacts. For classification models, the default naming of the compiled artifacts is model structure name_recurrence mode dataset abbreviation_batch size.
#### Execute the following commands in the tools/ directory to compile and run the regular versions of the models:
  ```Python
python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2dm_itout-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1cd_itout-b64x1-cifar10dvs --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif3flif2dg_itout-b16x1-dvsgesture --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1dg_itout-b16x1-dvsgesture --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config fasttext-b16x1-imdb --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config resnetlif18-itout-b20x4-16-jester --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif3fc3lc_itout-b16x1-luna16cls --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config resnetlif50-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config resnetlif18-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif3flif2rg_itout-b16x1-rgbgesture --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config resnetlif18-itout-b64x4-esimagenet --checkpoint latest.pth --use_lyngor 1 --use_legacy 0

python3 test.py --config clifplus3fc3dm_itout-b16x1-dvsmnist --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clifplus5fc2cd_itout-b64x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clifplus3flifplus2dg_itout-b16x1-dvsgesture --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config fasttextlifplus-b16x1-imdb --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clifplus3fc3lc_itout-b16x1-luna16cls --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clifplus3flifplus2rg_itout-b16x1-rgbgesture --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
  ```
#### Execute the following commands in the tools/ directory to compile and run models with batch size > 1 (using 2 as an example):
Note: For batch size > 1, use V1 compiler. The default batch size is 1, and supported batch sizes are 2, 4, 7, 14, 28. Smaller models can support up to 28batch. larger models may report the error 'Models with unequal numbers of batch and n_slice are not supported yet. Please reduce the batchsize number!' due to the compiler's underlying cutoff. please reduce the batchsize value!
```Python
python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif5fc2dm_itout-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif7fc1cd_itout-b64x1-cifar10dvs --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif3flif2dg_itout-b16x1-dvsgesture --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif7fc1dg_itout-b16x1-dvsgesture --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config resnetlif18-itout-b20x4-16-jester --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif3fc3lc_itout-b16x1-luna16cls --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config resnetlif50-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config resnetlif18-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config clif3flif2rg_itout-b16x1-rgbgesture --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
python3 test.py --config resnetlif18-itout-b64x4-esimagenet --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --v 1 --b 2
  ```
#### Execute the following commands in the tools/ directory to compile and run ResNet Lite versions of the models:
  ```Python
  python3 test.py --config resnetlif18-lite-itout-b20x4-16-jester --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config resnetlif50-lite-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config resnetlif18-lite-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
  ```

#### Execute the following commands in the tools/ directory to compile and run models with on-chip load/save:
```Python
python3 test_glb.py --config clif5fc2dm_itout-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config fasttext-b16x1-imdb --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config resnetlif18-itout-b20x4-16-jester --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config clif3fc3lc_itout-b16x1-luna16cls --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --mode 1
  ```

#### Execute the following commands in the tools/ directory to compile and run iter in versions of the models
```Python
python3 test.py --config resnetlif50-it-b16x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1cd_it-b64x1-cifar10dvs --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1dg_it-b16x1-dvsgesture --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2dm_it-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
```
  
### 2. Compilation without APU Inference.If you only want to compile without performing APU inference, add --c 1 to the command. Example:
  ```Python
  python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0 --c 1
  ```
Note: --v indicates compiling the version, such as v1 or v0, --b represents the batch size,  --c means to compile only，and --device set apu chip id.
### 3. Model Partitioning
#### Execute the following command in the tools/ directory to compile model partitioning:
  ```Python
  python3 complie_for_mp.py --config resnetlif50-itout-b8x1-cifar10dvs_mp  --checkpoint latest.pth
  ```
#### Execute the following command in the tools/ directory for time division with multi-card pipelining:
  ```Python
  python3 apuinfer_mutidevice.py --config resnetlif50-itout-b8x1-cifar10dvs
  ```
#### Execute the following command in the tools/ directory for model division with multi-card pipelining:
  ```Python
  python3 apuinfer_mutidevice.py --config resnetlif50-itout-b8x1-cifar10dvs_mp
  ```

### 4. Spike Encoding of MNIST Dataset
#### Execute the following commands in tools/ to perform APU inference using three encoding methods: rate, time, and population.
  ```Python
  python3 test.py --config clif3fc3mnrate_itout-b128x1-mnist --checkpoint latest_rate.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config clif3fc3mntime_itout-b128x1-mnist --checkpoint latest_time.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config clif3fc3mnpop_itout-b64x1-mnist --checkpoint latest_pop.pth --use_lyngor 1 --use_legacy 0
  ```

### 5. If you need to add the post_mode compilation option in test.py, you can refer to the following example
  ```Python
  python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0 --post_mode 1005
  ```

### 6. SpikeformerV2 classification on ImageNet 
#### Execute the following commands in spikeformerv2/classification/lynchip/ to perform APU inference
```Python
  python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0 --compile True
  python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0
  ```

### 7. Train
#### Execute the following commands in the tools/ directory to  train the models:
  ```Python
  python3 train.py --config clif3fc3dm_itout-b16x1-dvsmnist
  python3 train.py --config clif5fc2dm_itout-b16x1-dvsmnist
  python3 train.py --config clif3flif2dg_itout-b16x1-dvsgesture
  python3 train.py --config clif7fc1dg_it-b16x1-dvsgesture
  python3 train.py --config clif5fc2cd_itout-b64x1-cifar10dvs
  python3 train.py --config clif7fc1cd_itout-b64x1-cifar10dvs
  python3 train.py --config resnetlif18-itout-b8x1-cifar10dvs
  python3 train.py --config resnetlif18-lite-itout-b8x1-cifar10dvs
  python3 train.py --config fasttext-b16x1-imdb
  python3 train.py --config resnetlif18-itout-b20x4-16-jester
  python3 train.py --config clif3fc3lc_itout-b16x1-luna16cls
  python3 train.py --config clif3flif2rg_itout-b16x1-rgbgesture
  python3 train.py --config resnetlif18-itout-b64x4-esimagenet
  python3 train.py --config clif3fc3mnrate_itout-b128x1-mnist
  python3 train.py --config clif3fc3mntime_itout-b128x1-mnist
  python3 train.py --config clif3fc3mnpop_itout-b64x1-mnist
  python3 train.py --config resnetlif50-it-b16x1-cifar10dvs
  ```
