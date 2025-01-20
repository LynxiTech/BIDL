


## 在``src/main/``目录下执行如下命令进行模型的编译和运行
### 1. 执行此命令，进行模型的gpu/cpu推理
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 0 --use_legacy 0 --c 0
  ```

### 2. 执行此命令，进行apu推理
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 0 --use_legacy 1 --c 0
  ``` 
- 注：use_legacy为1时，use_lyngor和c这两个参数的值无效，但建议都设为0
- 执行此命令时要确保placerecog/model_config/下有四个模块的编译生成物，否则要先执行命令3或命令4
- 如果要统计apu的帧率，需要使用此条命令

### 3. 执行此命令，进行模型的编译和apu推理
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 1 --use_legacy 0 --c 0
  ```

### 4. 执行此命令，只进行模型的编译，不进行apu推理
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 1 --use_legacy 0 --c 1
  ``` 
- 注：只进行模型的编译时，c的值设为1，其他情况都设为0