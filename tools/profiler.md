### lyngor profiler analysis example
# step 1
python3 test.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 0 --c 1 --profiler True #只编译模型，生成debug.json
# step 2
profile_log "python3 test.py --config clif5fc2cd_itout-b64x1-cifar10dvs --checkpoint best_clif5.pth --use_lyngor 1 --use_legacy 1" --config ../model_files/classification/Clif5fc2cd_itoutCd/Net_0/apu_0/debug.json
# step 3
profile_parse -i output -g ../model_files/classification/Clif5fc2cd_itoutCd/ -o ./clif5fc2cd_itout-profile-remove_all_load_save --all

