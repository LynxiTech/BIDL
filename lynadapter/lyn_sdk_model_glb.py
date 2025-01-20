# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.

import sys
import pylynchipsdk as sdk
import numpy as np
import os
import threading
from queue import Queue
from collections import deque
from datetime import datetime

dtype_dict = {
            sdk.lyn_data_type_t.DT_INT8: np.byte,
            sdk.lyn_data_type_t.DT_UINT8: np.ubyte,
            sdk.lyn_data_type_t.DT_INT16: np.int16,
            sdk.lyn_data_type_t.DT_UINT16: np.uint16,
            sdk.lyn_data_type_t.DT_INT32: np.int32,
            sdk.lyn_data_type_t.DT_UINT32: np.uint32,
            sdk.lyn_data_type_t.DT_INT64: np.int64,
            sdk.lyn_data_type_t.DT_UINT64: np.uint64,
            sdk.lyn_data_type_t.DT_FLOAT: np.float32,
            sdk.lyn_data_type_t.DT_FLOAT16: np.half,
            sdk.lyn_data_type_t.DT_DOUBLE: np.float64
        }

def error_check(condition, log):
    if (condition):
        print("\n****** {} ERROR: {}".format(datetime.now(), log))
        os._exit(-1)

def error_check_handler(stream, errorMsg, params): 
    print("******* streamID : ", stream) 
    print("******* {} errorCode : {}".format(datetime.now(),errorMsg.errCode))
    error_check(1, "stream error")

SWAP_POOL_COUNT = 20

class buffer_pool:
    def __init__(self, elemsize, maxszie):
        self.isinit = False
        self.elemsize = elemsize
        self.maxszie = maxszie
        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        self.is_full = threading.Condition(self.mutex)
        self.queue = deque()
        for x in range(maxszie):
            addr, ret = sdk.lyn_malloc(elemsize)
            self.queue.append(addr)
        self.isinit = True

    def free_buffers(self):
        if not self.isinit:
            raise ValueError("buffer pool not init")
            
        with self.mutex :
            while self._qsize() < self.maxszie:
                self.is_full.wait()
            for addr in self.queue:
                sdk.lyn_free(addr)
            self.isinit = False

    def full(self):
        if not self.isinit:
            raise ValueError("buffer pool not init")
        with self.mutex :
            return 0 < self.maxszie <= self._qsize()

    def empty(self):
        if not self.isinit:
            raise ValueError("buffer pool not init")
        with self.mutex :
            return not self._qsize()

    def push(self, item):
        if not self.isinit:
            raise ValueError("buffer pool not init")
        if self.full():
            raise ValueError("buffer pool is full")

        with self.mutex :
            self.queue.append(item)
            self.not_empty.notify()
            if self._qsize() >= self.maxszie:
                self.is_full.notify()

    def pop(self):
        if not self.isinit:
            raise ValueError("buffer pool not init")

        with self.mutex :
            while self._qsize() == 0 :
                self.not_empty.wait()
            item = self.queue.popleft()
            return item

    def qsize(self):
        with self.mutex:
            return self._qsize()

    def _qsize(self):
        return len(self.queue)

RR = []
def get_result_callback(params):
    total_output_data_len = params[0]
    apu_in_pool = params[3]
    apu_in_buffer = params[4]
    apu_out_pool = params[1]
    apu_out_buffer = params[2]
    batch_size = params[5]

    data = np.ones(total_output_data_len*batch_size, dtype=np.byte)
    data_ptr = sdk.lyn_numpy_to_ptr(data)
    ret = sdk.lyn_memcpy(data_ptr, apu_out_buffer, total_output_data_len*batch_size,
                                sdk.lyn_memcpy_dir_t.ServerToClient)
   
    output = sdk.lyn_ptr_to_numpy(data_ptr, (1, int(total_output_data_len*batch_size/4)),
                                    sdk.lyn_data_type_t.DT_FLOAT)

    RR.append(output) 
    apu_out_pool.push(apu_out_buffer)
    apu_in_pool.push(apu_in_buffer)   


    return 0


class ApuRun:   
    def __init__(self, device_id, model_path,timestep):
        try:
            self.context, ret = sdk.lyn_create_context(device_id)
            error_check(ret != 0, "lyn_create_context")
            sdk.lyn_set_current_context(self.context)
            ret = sdk.lyn_register_error_handler(error_check_handler)
            error_check(ret != 0, "lyn_register_error_handler")
            self.stream, ret = sdk.lyn_create_stream()
            error_check(ret != 0, "lyn_create_stream")
            
            self.model, ret = sdk.lyn_load_model(model_path)
            error_check(ret != 0, "lyn_load_model")
            self.model_desc, ret = sdk.lyn_model_get_desc(self.model)
            error_check(ret != 0, "lyn_model_get_desc")

            self.total_output_data_len = self.model_desc.outputDataLen

            self.batch_size = self.model_desc.inputTensorAttrArray[0].batchSize
            #print("self.total_output_data_len " , self.total_output_data_len,self.batch_size)
            self.host_out_ptr = sdk.c_malloc(self.total_output_data_len*self.batch_size)
            error_check(ret != 0, "c_malloc")
            # print("befor lyn malloc device id ", device_id)
            self.dev_out_ptr, ret = sdk.lyn_malloc(self.total_output_data_len*self.batch_size)
            error_check(ret != 0, "lyn_malloc")
            # print("after lyn malloc device id ", device_id)
            self.total_input_data_len = self.model_desc.inputDataLen  # * model_desc.inputTensorAttrArray[0].batchSize
            #print("self.total_input_data_len " , self.total_input_data_len)
            self.dev_ptr, ret = sdk.lyn_malloc(self.total_input_data_len*self.batch_size)
            # print("last malloc self.dev_ptr ",self.dev_ptr, self.total_input_data_len,device_id)
            error_check(ret != 0, "lyn_malloc")

            sdk.lyn_model_reset_async(self.stream, self.model)
            self.model_path = model_path
            self.apu_in_pool = buffer_pool(self.total_input_data_len*self.batch_size, SWAP_POOL_COUNT)
            self.apu_out_pool = buffer_pool(self.total_output_data_len*self.batch_size, SWAP_POOL_COUNT)
            self.in_array=[i for i in range(5000)]
            self.timestep = timestep
            self.run_time = 0
        except:
            self.release()
            raise Exception('failed')
    
    def load_model(self, model_path):
        if hasattr(self, 'model'):
            sdk.lyn_set_current_context(self.context)
            sdk.lyn_unload_model(self.model)   
        
        self.model, ret = sdk.lyn_load_model(model_path)
        error_check(ret != 0, "lyn_load_model")
        self.model_desc, ret = sdk.lyn_model_get_desc(self.model)
        error_check(ret != 0, "lyn_model_get_desc")

        self.total_output_data_len = self.model_desc.outputDataLen
        self.batch_size = self.model_desc.inputTensorAttrArray[0].batchSize

        sdk.lyn_model_reset_async(self.stream, self.model)
        
        self.apu_in_pool = buffer_pool(self.total_input_data_len*self.batch_size, SWAP_POOL_COUNT)
        self.apu_out_pool = buffer_pool(self.total_output_data_len*self.batch_size, SWAP_POOL_COUNT)

    def load_data(self, data):
        """
        preconditions: input tensors have same dtype; all input tensors in input_file by order
        """
        sdk.lyn_set_current_context(self.context) 
        dtype_dict = {
            sdk.lyn_data_type_t.DT_INT8: np.byte,
            sdk.lyn_data_type_t.DT_UINT8: np.ubyte,
            sdk.lyn_data_type_t.DT_INT16: np.int16,
            sdk.lyn_data_type_t.DT_UINT16: np.uint16,
            sdk.lyn_data_type_t.DT_INT32: np.int32,
            sdk.lyn_data_type_t.DT_UINT32: np.uint32,
            sdk.lyn_data_type_t.DT_INT64: np.int64,
            sdk.lyn_data_type_t.DT_UINT64: np.uint64,
            sdk.lyn_data_type_t.DT_FLOAT: np.float32,
            sdk.lyn_data_type_t.DT_FLOAT16: np.half,
            sdk.lyn_data_type_t.DT_DOUBLE: np.float64
        }
        array = data.astype(dtype_dict[self.model_desc.inputTensorAttrArray[0].dtype])
        host_ptr = sdk.lyn_numpy_to_ptr(array)
        sdk.lyn_memcpy(self.dev_ptr, host_ptr, self.total_input_data_len*self.batch_size,
                       sdk.lyn_memcpy_dir_t.ClientToServer)  # todo repeat

        assert all([self.dev_ptr, host_ptr]), "prepare_input_ptr fail"  
    
    def run(self,data_loader,d_idx):
        import copy
        apu_in_buffer = self.apu_in_pool.pop()
        apu_out_buffer = self.apu_out_pool.pop()       
        
        for ti in range(self.timestep):
            data_img = data_loader[:, ti, ...]
            data_img = np.array(data_img,dtype=dtype_dict[self.model_desc.inputTensorAttrArray[0].dtype])
            
            self.in_array[(d_idx*self.timestep+ti)%5000] = copy.deepcopy(data_img)
            host_ptr = sdk.lyn_numpy_to_ptr(self.in_array[(d_idx*self.timestep+ti)%5000])

            # self.in_array[self.run_time] = copy.deepcopy(data_img)
            # host_ptr = sdk.lyn_numpy_to_ptr(self.in_array[self.run_time])

            sdk.lyn_set_current_context(self.context)
            sdk.lyn_memcpy_async(self.stream,apu_in_buffer, host_ptr, self.total_input_data_len*self.batch_size,
                    sdk.lyn_memcpy_dir_t.ClientToServer)  # todo repeat
            
            if ti == 0:
                ret = sdk.lyn_model_reset_async(self.stream, self.model)
                if ret != 0:
                    print("lyn_model_reset_async fail 0x%x" % ret)
                    return 1                        
            ret = sdk.lyn_execute_model_async(self.stream, self.model, apu_in_buffer, apu_out_buffer,
                            self.batch_size)
            assert ret == 0, 'lyn_execute_model_async fail 0x{:x}'.format(ret)
            
            if ti==(self.timestep-1):
                ret = sdk.lyn_stream_add_callback(self.stream, get_result_callback,[self.total_output_data_len,self.apu_out_pool,apu_out_buffer,self.apu_in_pool,apu_in_buffer,self.batch_size] )
                assert ret == 0, 'lyn_stream_add_callback fail 0x{:x}'.format(ret) 
        self.run_time +=1

        return 0
    
    def run_glb(self,data_loader,d_idx):
        if self.run_time >= 1:
            self.load_model(self.model_path )
        import copy
        apu_in_buffer = self.apu_in_pool.pop()
        apu_out_buffer = self.apu_out_pool.pop()       
        
        for ti in range(self.timestep):
            data_img = data_loader[:, ti, ...]
            data_img = np.array(data_img,dtype=dtype_dict[self.model_desc.inputTensorAttrArray[0].dtype])
            
            self.in_array[(d_idx*self.timestep+ti)%5000] = copy.deepcopy(data_img)
            host_ptr = sdk.lyn_numpy_to_ptr(self.in_array[(d_idx*self.timestep+ti)%5000])

            # self.in_array[self.run_time] = copy.deepcopy(data_img)
            # host_ptr = sdk.lyn_numpy_to_ptr(self.in_array[self.run_time])

            sdk.lyn_set_current_context(self.context)
            sdk.lyn_memcpy_async(self.stream,apu_in_buffer, host_ptr, self.total_input_data_len*self.batch_size,
                    sdk.lyn_memcpy_dir_t.ClientToServer)  # todo repeat
            
            if ti == 0:
                ret = sdk.lyn_model_reset_async(self.stream, self.model)
                if ret != 0:
                    print("lyn_model_reset_async fail 0x%x" % ret)
                    return 1                        
            ret = sdk.lyn_execute_model_async(self.stream, self.model, apu_in_buffer, apu_out_buffer,
                            self.batch_size)
            assert ret == 0, 'lyn_execute_model_async fail 0x{:x}'.format(ret)
            
            if ti==(self.timestep-1):
                ret = sdk.lyn_stream_add_callback(self.stream, get_result_callback,[self.total_output_data_len,self.apu_out_pool,apu_out_buffer,self.apu_in_pool,apu_in_buffer,self.batch_size] )
                assert ret == 0, 'lyn_stream_add_callback fail 0x{:x}'.format(ret) 
        self.run_time +=1
        self.apu_in_pool.free_buffers()   
        self.apu_out_pool.free_buffers()

        return 0


    def run_single(self, timestep):
        
        if timestep == 0:
            sdk.lyn_set_current_context(self.context)
            ret = sdk.lyn_model_reset_async(self.stream, self.model)
            if ret != 0:
                print("lyn_model_reset_async fail 0x%x" % ret)
                return 1
        sdk.lyn_set_current_context(self.context)
        ret = sdk.lyn_execute_model_async(self.stream, self.model, self.dev_ptr, self.dev_out_ptr,
                                          self.batch_size)
        assert ret == 0, 'lyn_execute_model_async fail 0x{:x}'.format(ret)
        sdk.lyn_set_current_context(self.context)

        ret = sdk.lyn_synchronize_stream(self.stream)
        assert ret == 0, 'lyn_synchronize_stream fail 0x{:x}'.format(ret)
        return 0

    def run_tail(self):

        sdk.lyn_set_current_context(self.context)

        ret = sdk.lyn_execute_model_async(self.stream, self.model, self.dev_ptr, self.dev_out_ptr,
                                          self.batch_size)
        assert ret == 0, 'lyn_execute_model_async fail 0x{:x}'.format(ret)
        sdk.lyn_set_current_context(self.context)

        ret = sdk.lyn_synchronize_stream(self.stream)
        assert ret == 0, 'lyn_synchronize_stream fail 0x{:x}'.format(ret)
        return 0

    def get_output(self):
        sdk.lyn_set_current_context(self.context)
        ret = sdk.lyn_memcpy(self.host_out_ptr, self.dev_out_ptr, self.total_output_data_len*self.batch_size,
                             sdk.lyn_memcpy_dir_t.ServerToClient)
        error_check(ret != 0, "lyn_memcpy fail")

        # output = sdk.lyn_ptr_to_numpy(self.host_out_ptr, (1, 45089), sdk.lyn_data_type_t.DT_FLOAT)
        output = sdk.lyn_ptr_to_numpy(self.host_out_ptr, (1, int(self.total_output_data_len*self.batch_size / 4)),
                                      sdk.lyn_data_type_t.DT_FLOAT)
        return output

    def syn_op(self):
        sdk.lyn_set_current_context(self.context)
        ret = sdk.lyn_synchronize_stream(self.stream)
        assert ret == 0, 'lyn_synchronize_stream fail 0x{:x}'.format(ret)    
         
        return RR

    def release(self):
        if hasattr(self, 'host_out_ptr'):
            sdk.lyn_set_current_context(self.context)
            sdk.c_free(self.host_out_ptr)
        if hasattr(self, 'dev_out_ptr'):
            sdk.lyn_set_current_context(self.context)
            sdk.lyn_free(self.dev_out_ptr)
        if hasattr(self, 'dev_ptr'):
            sdk.lyn_set_current_context(self.context)
            sdk.lyn_free(self.dev_ptr)
        if hasattr(self, 'model'):
            sdk.lyn_set_current_context(self.context)
            sdk.lyn_unload_model(self.model)
        if hasattr(self, 'stream'):
            sdk.lyn_set_current_context(self.context)
            sdk.lyn_destroy_stream(self.stream)
        if hasattr(self, 'context'): 
            sdk.lyn_set_current_context(self.context)          
            sdk.lyn_destroy_context(self.context)
    
    def __del__(self):
        self.release()