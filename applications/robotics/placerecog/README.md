


## Compile and run the model in the src/main/ directory using the following commands:
### 1. Execute this command for GPU/CPU inference of the model
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 0 --use_legacy 0 --c 0
  ```

### 2. Execute this command for APU inference
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 0 --use_legacy 1 --c 0
  ``` 
- Note: When use_legacy is set to 1, the values of use_lyngor and c parameters are ignored, but it is recommended to 
  set them  both to 0.
- Make sure that there are compiled artifacts for all four modules under placerecog/model_config/ directory before 
  executing this command. Otherwise, execute command 3 or command 4 first.
- If you want to measure the frame rate of APU, use this command.

### 3. Execute this command to compile the model and perform APU inference
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 1 --use_legacy 0 --c 0
  ```

### 4. Execute this command to only compile the model without performing APU inference
  ```Python
  python3 main_inference.py --config_file ../config/room_setting.ini --use_lyngor 1 --use_legacy 0 --c 1
  ``` 
- Note: When only compiling the model, set the value of c to 1, and set it to 0 for other cases.





