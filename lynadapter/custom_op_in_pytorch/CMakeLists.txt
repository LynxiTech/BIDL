# make custom_ops.so command(depends on libtorch):
#   cmake -DCMAKE_PREFIX_PATH=your/libtorch/path ..
#   make -j

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)  # 提高了版本要求以支持更多的特性  
project(custom_ops)  
  
set(CMAKE_PREFIX_PATH "/path/to/libtorch" CACHE PATH "Path to libtorch")  
set(PYTORCH_VERSION "")  

execute_process(  
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_pytorch_version.py  
    OUTPUT_VARIABLE PYTORCH_VERSION_RAW  
    OUTPUT_STRIP_TRAILING_WHITESPACE  
)  
message(STATUS "Raw PyTorch version output: ${PYTORCH_VERSION_RAW}") 
string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" PYTORCH_MAJOR_VERSION "${PYTORCH_VERSION_RAW}")  
find_package(Torch REQUIRED)  
  
add_library(custom_ops SHARED  
    custom_op_myLif.cpp  
    custom_op_Rand.cpp  
)  
  
if(PYTORCH_MAJOR_VERSION GREATER 1)  
    set_property(TARGET custom_ops PROPERTY CXX_STANDARD 17)  
    message(STATUS "Setting C++ standard to 17 for PyTorch version ${PYTORCH_VERSION}")  
else()  
    set_property(TARGET custom_ops PROPERTY CXX_STANDARD 14)  
    message(WARNING "Using PyTorch version ${PYTORCH_VERSION} which may not be fully compatible with C++14. It is highly recommended to use C++17 with PyTorch.")  
endif()  
set_property(TARGET custom_ops PROPERTY CXX_STANDARD_REQUIRED ON) 
  
target_link_libraries(custom_ops "${TORCH_LIBRARIES}")   
target_compile_definitions(custom_ops PRIVATE custom_ops_EXPORTS) 


