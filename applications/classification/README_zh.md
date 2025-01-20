## 执行指令
### 1. 编译并运行已有的模型
Note：所有模型的编译生成物默认存放路径为项目根目录下的./model_files中，具体为./model_files/app所属类别/编译生成物，
对于分类模型来说，编译生成物的默认命名为模型结构名_循环方式数据集简称_batchsize    
#### 在``tools/``目录下执行如下命令，进行常规版本的编译和运行
  ```Python
python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2dm_itout-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1cd_itout-b64x1-cifar10dvs --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif3flif2dg_itout-b16x1-dvsgesture --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1dg_itout-b16x1-dvsgesture --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config fasttext-b16x1-imdb --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config resnetlif18-itout-b20x4-16-jester --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
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
#### 在``tools/``目录下执行如下命令，进行batchsize>1版本模型的编译和运行(以2batch为例)
Note: batchsize>1 需使用V1编译器。batchsize默认为1，支持的batchsize数为2, 4, 7, 14, 28。小模型最高可支持28batch。模型较大时由于编译器底层切分，可能出现报错'Models with unequal numbers of batch and n_slice are not supported yet. Please reduce the batchsize number！'。请降低batchsize值！
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
#### 在``tools/``目录下执行如下命令，进行resnet lite版本的编译和运行
  ```Python
  python3 test.py --config resnetlif18-lite-itout-b20x4-16-jester --checkpoint latest.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config resnetlif50-lite-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config resnetlif18-lite-itout-b8x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
  ```
#### 在``tools/``目录下执行如下命令，进行片上load save版本模型的编译和运行
```Python
python3 test_glb.py --config clif5fc2dm_itout-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config fasttext-b16x1-imdb --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config resnetlif18-itout-b20x4-16-jester --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --mode 1
python3 test_glb.py --config clif3fc3lc_itout-b16x1-luna16cls --checkpoint latest.pth --use_lyngor 1 --use_legacy 0 --mode 1
  ```

#### 在``tools/``目录下执行如下命令，进行内循环版本的编译和运行
```Python
python3 test.py --config resnetlif50-it-b16x1-cifar10dvs --checkpoint best.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2cd_it-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1cd_it-b64x1-cifar10dvs --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif7fc1dg_it-b16x1-dvsgesture --checkpoint best_clif7.pth --use_lyngor 1 --use_legacy 0
python3 test.py --config clif5fc2dm_it-b16x1-dvsmnist --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0
```
  
### 2. 如果只编译，不进行apu推理，需要加上--c 1，示例如下
  ```Python
  python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0 --c 1
  ```
Note: --v 表示编译version，v1或者v0，--b 表示batchsize大小， --c表示只编译 --device 设置apu的chip id
### 3. 模型切分
#### 在``tools/``目录下执行如下命令，进行模型切分编译
  ```Python
  python3 complie_for_mp.py --config resnetlif50-itout-b8x1-cifar10dvs_mp  --checkpoint latest.pth
  ```
#### 在``tools/``目录下执行如下命令，进行时间切分多卡流水并行
  ```Python
  python3 apuinfer_mutidevice.py --config resnetlif50-itout-b8x1-cifar10dvs
  ```
#### 在``tools/``目录下执行如下命令，进行模型切分多卡流水并行
  ```Python
  python3 apuinfer_mutidevice.py --config resnetlif50-itout-b8x1-cifar10dvs_mp
  ```

### 4. MNIST数据集的脉冲编码方式
#### 在``tools/``执行一下指令，分别进行rate、time和population三种编码方式的apu推理
  ```Python
  python3 test.py --config clif3fc3mnrate_itout-b128x1-mnist --checkpoint latest_rate.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config clif3fc3mntime_itout-b128x1-mnist --checkpoint latest_time.pth --use_lyngor 1 --use_legacy 0
  python3 test.py --config clif3fc3mnpop_itout-b64x1-mnist --checkpoint latest_pop.pth --use_lyngor 1 --use_legacy 0
  ```
### 5. 如果需要在test.py中加上post_mode的编译选项，可以参考如下示例
  ```Python
  python3 test.py --config clif3fc3dm_itout-b16x1-dvsmnist --checkpoint best_clif3.pth --use_lyngor 1 --use_legacy 0 --post_mode 1005
  ```

### 6. Spikeformerv2分类imagenet
#### 在``spikeformerv2/classification/lynchip``执行一下指令，进行apu推理
  ```Python
  python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0 --compile True
  python3 lynxi_inference.py --batch_size 1 --model metaspikformer_8_512  --time_steps 4 --device apu:0
  ```

### 7. 训练
#### 在``tools/``目录下执行如下命令，进行模型的训练
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
