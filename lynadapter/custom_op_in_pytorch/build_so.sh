#!/bin/bash

# build custom op


echo "file：$0"
echo "param 1：$1 Whether to compile and generate a dynamic link library for custom operators"
echo "param 2：$2 The path of the PyTorch library used by the custom operator (make sure that the directory hierarchy "/lib/libtorch.so" exists in this path)"


# your_libtorch_path="~/test/pytorch-master/libtorch"
your_libtorch_path=$2
your_libtorch_path_lib="$your_libtorch_path/lib"



# To add the lib path of libtorch to the environment variable, 
# you need to use a shell script for compiling and dynamically linking the custom operator library.
export LD_LIBRARY_PATH=$your_libtorch_path_lib:$LD_LIBRARY_PATH
filename="libcustom_ops"
msgPrefix="[INFO]"


if [ ${1:-1} -eq 1 ]
then
    echo "$msgPrefix start build custom op so"
    if [ -d "build" ];then
        echo "$msgPrefix build folder already exist, delete the old one"
        rm -r build
        mkdir build
    else
        echo "$msgPrefix build folder not exist, build a new one"
        mkdir build
    fi

    cd build

    if [ -f "$filename.so" ];then
        echo "$msgPrefix $filename.so exist, delete old one"
        rm -$filename.so
    fi

    echo "$msgPrefix start cmake"
    cmake -DCMAKE_PREFIX_PATH=$your_libtorch_path ..
    echo "$msgPrefix finish cmake"

    echo "$msgPrefix start make"
    make -j32
    echo "$msgPrefix finish make"

    if [ -f "$filename.so" ];then
        echo "$msgPrefix $filename.so generate"
        echo "$msgPrefix build success"
    else
        echo "$msgPrefix $filename.so not generate"
        echo "$msgPrefix build fail"
    fi

    cd ..
else
    echo "$msgPrefix not build so"
fi

echo "$msgPrefix finish build custom op so"
