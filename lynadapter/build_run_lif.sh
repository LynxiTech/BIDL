#!/bin/bash

# build custom so and run test_custom_op

# Passing arguments from outside $0 $1 $2
echo "file：$0"
echo "param 1：$1 build custom op so"
echo "param 2：$2 run test_custom_op"

# You need to configure the path for your libtorch - 
# Make sure that the directory hierarchy /lib/libtorch.so exists under the path TORCH_LIBRARIES.
torch_module_location=`pip3 show torch |grep Location |awk '{print $2}'`
# your_libtorch_path_lib="$your_libtorch_path/lib"
your_libtorch_path="${torch_module_location}/torch" # Absolute path is required


out_msg="path of Pytorch lib : $your_libtorch_path"
echo -e "\033[31m$out_msg\033[0m"
# export LD_LIBRARY_PATH=$your_libtorch_path_lib:$LD_LIBRARY_PATH
# export TORCH_LIBRARIES=$your_libtorch_path_lib

msgPrefix="[INFO]"


if [ ${1:-1} -eq 1 ]
then
    cd ./custom_op_in_pytorch/
    source build_so.sh 1 $your_libtorch_path
    cd ..
else
    echo "$msgPrefix not build so"
fi


# if [ ${2:-1} -eq 1 ]
# then
#     echo "$msgPrefix start run test_custom_op"
#     echo "[===]$LD_LIBRARY_PATH"
#     echo "[===]$TORCH_LIBRARIES"
#     python3.6 00000.py
#     echo "$msgPrefix finish run test_custom_op"
# else
#     echo "$msgPrefix not run test_custom_op"
# fi

# echo "$msgPrefix finish build custom op so"
