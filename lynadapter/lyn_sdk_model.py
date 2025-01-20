# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


import os
import pylynchipsdk as sdk
from datetime import datetime
import copy
import numpy as np
from queue import Queue

# {SDK data type: [corresponding numpy data type, number of bytes]}
dtype_dict = {
            sdk.lyn_data_type_t.DT_INT8: [np.byte, 1],
            sdk.lyn_data_type_t.DT_UINT8: [np.ubyte, 1],
            sdk.lyn_data_type_t.DT_INT16: [np.int16, 2],
            sdk.lyn_data_type_t.DT_UINT16: [np.uint16, 2],
            sdk.lyn_data_type_t.DT_INT32: [np.int32, 4],
            sdk.lyn_data_type_t.DT_UINT32: [np.uint32, 4],
            sdk.lyn_data_type_t.DT_INT64: [np.int64, 8],
            sdk.lyn_data_type_t.DT_UINT64: [np.uint64, 8],
            sdk.lyn_data_type_t.DT_FLOAT: [np.float32, 4],
            sdk.lyn_data_type_t.DT_FLOAT16: [np.half, 2],
            sdk.lyn_data_type_t.DT_DOUBLE: [np.float64, 8],
        }

C2S = sdk.lyn_memcpy_dir_t.ClientToServer
S2C = sdk.lyn_memcpy_dir_t.ServerToClient

def error_check(condition, log):
    if (condition):
        print("\n****** {} ERROR: {}".format(datetime.now(), log))
        os._exit(-1)

def error_check_handler(stream, errorMsg, params): 
    print("******* streamID : ", stream) 
    print("******* {} errorCode : {}".format(datetime.now(),errorMsg.errCode))
    error_check(1, "stream error")


def split_array_by_lengths(arr, segment_lengths, axis=0):
    # Calculate the segmentation position
    split_points = []
    split_position = 0
    for length in segment_lengths[:]:
        split_position += length
        split_points.append(split_position)

    # Divide the array into n segments.
    segments = np.split(arr, split_points, axis=axis)
    
    # Check if the last paragraph needs to be adjusted in length.
    last_segment_length = arr.shape[axis] - sum(segment_lengths[:-1])
    if last_segment_length != segment_lengths[-1]:
        index = [slice(None)] * arr.ndim
        index[axis] = slice(last_segment_length)
        segments[-1] = arr[tuple(index)]

    return segments

def calculate_product(lst):
    product = 1
    for num in lst:
        product *= num
    return product

# Get the inference results into self.output_list
# self.output_list[Number of datasets][Number of model outputs][Time steps][Model output]
def get_result_callback(params):
    self = params[0]
    apuinbuf = params[1]
    apuoutbuf = params[2]

    ret = 0
    hostoutbuf = sdk.c_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize'] * self.time_steps)
    # ret |= sdk.lyn_memcpy(hostoutbuf, sdk.lyn_addr_seek(apuoutbuf, self.modelDict['outputdatalen'] * (self.time_steps - 1)), self.modelDict['outputdatalen'], S2C)
    ret |= sdk.lyn_memcpy(hostoutbuf, apuoutbuf, self.modelDict['outputdatalen'] * self.modelDict['batchsize'] * self.time_steps, S2C)
    error_check(ret != 0, "lyn_memcpy")
    # outputArrary = copy.deepcopy(sdk.lyn_ptr_to_numpy(hostoutbuf,(self.time_steps,self.modelDict['batchsize'],int(self.modelDict['outputdatalen']/dtype_dict[self.modelDict['outputdatatype']][1]),),\
    #     dtype_dict[self.modelDict['outputdatatype']][0]))
    outputArrary = sdk.lyn_ptr_to_numpy(hostoutbuf, (self.time_steps, self.modelDict['batchsize'], int(self.modelDict['outputdatalen'] / dtype_dict[self.modelDict['outputdatatype']][1]), ),\
        dtype_dict[self.modelDict['outputdatatype']][0])

    segment_lengths = []
    output = []
    for i in range(self.modelDict['outputnum']):
        segment_lengths.append(calculate_product(self.modelDict['outputshape'][i]))
    spilt_arr = split_array_by_lengths(outputArrary, segment_lengths, axis=-1)
    for i in range(self.modelDict['outputnum']):
        if self.modelDict['batchsize'] == 1:
            output.append(spilt_arr[i].reshape((self.time_steps,) + tuple(self.modelDict['outputshape'][i][:])))
        else:
            output.append(spilt_arr[i].reshape((self.time_steps, self.modelDict['batchsize']) + tuple(self.modelDict['outputshape'][i][1:])))

    self.output_list.insert(self.run_times, output)  

    # for b in range(self.modelDict['batchsize']):
    if self.propressbar is not None:
        self.propressbar.update()

    self.apuInPool.put(apuinbuf)
    self.apuOutPool.put(apuoutbuf)
    sdk.c_free(hostoutbuf)

    return ret


# for dataset 
class ApuRun:
    def __init__(self, apu_device, apu_model_path, time_steps=1):
        self.apu_device = apu_device
        self.time_steps = time_steps
        self.input_list_len = 3
        self.input_list = [None] * self.input_list_len
        self.input_ptr_list = [None] * self.input_list_len
        self.apuInPool = Queue(maxsize = self.input_list_len)
        self.apuOutPool = Queue(maxsize = self.input_list_len)
        self.output_list = []
        self.run_times = 0
        self.propressbar = None

        assert os.path.exists(apu_model_path), "apu model path not exists"
        ret = 0

        self._sdk_initialize()

        self.count, ret = sdk.lyn_get_device_count()
        print("chip count = ", self.count)
        error_check(ret != 0, "lyn_get_device_count")
        assert self.count > self.apu_device, "device id must be less device count"

        self.apu_model, ret = sdk.lyn_load_model(apu_model_path)
        error_check(ret != 0, "lyn_load_model")
        self._model_parse()

    def _sdk_initialize(self):
        ret = 0

        self.context, ret = sdk.lyn_create_context(self.apu_device)
        error_check(ret != 0, "lyn_create_context")

        ret = sdk.lyn_set_current_context(self.context)
        error_check(ret != 0, "lyn_set_current_context")

        ret = sdk.lyn_register_error_handler(error_check_handler)
        error_check(ret != 0, "lyn_register_error_handler")
    
        self.apu_stream_s, ret = sdk.lyn_create_stream()
        error_check(ret != 0, "lyn_create_stream")

        self.apu_stream_r, ret = sdk.lyn_create_stream()
        error_check(ret != 0, "lyn_create_stream")

        self.mem_reset_event, ret = sdk.lyn_create_event()
        error_check(ret != 0, "lyn_create_event")
    
    def _model_parse(self):
        ret = 0
        self.modelDict = {}
        model_desc, ret = sdk.lyn_model_get_desc(self.apu_model)
        error_check(ret != 0, "lyn_model_get_desc")
        
        self.modelDict['batchsize'] = model_desc.inputTensorAttrArray[0].batchSize
        self.modelDict['inputnum'] = len(model_desc.inputTensorAttrArray)
        inputshapeList = []
        for i in range(self.modelDict['inputnum']):
            inputDims = len(model_desc.inputTensorAttrArray[i].dims)
            inputShape = []
            for j in range(inputDims):
                inputShape.append(model_desc.inputTensorAttrArray[i].dims[j])
            inputshapeList.append(inputShape)
        self.modelDict['inputshape'] = inputshapeList    
        self.modelDict['inputdatalen'] = model_desc.inputDataLen
        self.modelDict['inputdatatype'] = model_desc.inputTensorAttrArray[0].dtype
        self.modelDict['outputnum'] = len(model_desc.outputTensorAttrArray)
        outputshapeList = []
        for i in range(self.modelDict['outputnum']):
            outputDims = len(model_desc.outputTensorAttrArray[i].dims)
            outputShape = []
            for j in range(outputDims):
                outputShape.append(model_desc.outputTensorAttrArray[i].dims[j])
            outputshapeList.append(outputShape)
        self.modelDict['outputshape'] = outputshapeList
        self.modelDict['outputdatalen'] = model_desc.outputDataLen
        self.modelDict['outputdatatype'] = model_desc.outputTensorAttrArray[0].dtype
        # print(self.modelDict)
        print('######## model informations ########')
        for key,value in self.modelDict.items():
            print('{}: {}'.format(key, value))
        print('####################################')

        for i in range(self.input_list_len):
            apuinbuf, ret = sdk.lyn_malloc(self.modelDict['inputdatalen'] * self.modelDict['batchsize'] * self.time_steps)
            self.apuInPool.put(apuinbuf)
            setattr(self, 'apuInbuf{}'.format(i), apuinbuf)

            apuoutbuf, ret = sdk.lyn_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize'] * self.time_steps) 
            self.apuOutPool.put(apuoutbuf)
            setattr(self, 'apuOutbuf{}'.format(i), apuoutbuf)

        self.hostOutbuf = sdk.c_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize'] * self.time_steps)
        
        for i in range(self.input_list_len):
            self.input_list[i] = np.zeros(self.modelDict['inputdatalen'] * self.modelDict['batchsize'] * self.time_steps//dtype_dict[self.modelDict['inputdatatype']][1], dtype = dtype_dict[self.modelDict['inputdatatype']][0])
            self.input_ptr_list[i] = sdk.lyn_numpy_to_ptr(self.input_list[i])
            
        self.dev_ptr, ret = sdk.lyn_malloc(self.modelDict['inputdatalen'] * self.modelDict['batchsize'])
        self.dev_out_ptr, ret = sdk.lyn_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize'])
        self.host_out_ptr = sdk.c_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize'])
 
    def run(self, img):
        assert isinstance(img, np.ndarray)

        currentInbuf = self.apuInPool.get(block=True)
        currentOutbuf = self.apuOutPool.get(block=True)

        ret = 0
        sdk.lyn_set_current_context(self.context)
        img = img.astype(dtype_dict[self.modelDict['inputdatatype']][0])
        i_id = self.run_times % self.input_list_len
        self.input_list[i_id][:] = img.flatten()
        # img_ptr, _ = sdk.lyn_numpy_contiguous_to_ptr(self.input_list[i_id])
        ret = sdk.lyn_memcpy_async(self.apu_stream_s, currentInbuf, self.input_ptr_list[i_id], self.modelDict['inputdatalen'] * self.modelDict['batchsize'] * self.time_steps, C2S)
        error_check(ret != 0, "lyn_memcpy_async")

        apuinbuf = currentInbuf
        apuoutbuf = currentOutbuf
        for step in range(self.time_steps):
            if step == 0:
                if self.run_times > 0:
                    sdk.lyn_stream_wait_event(self.apu_stream_s, self.mem_reset_event)
                ret = sdk.lyn_model_reset_async(self.apu_stream_s, self.apu_model)
                error_check(ret != 0, "lyn_model_reset_async")
            # ret = sdk.lyn_execute_model_async(self.apu_stream_s, self.apu_model, apuinbuf, apuoutbuf, self.modelDict['batchsize'])
            # error_check(ret!=0, "lyn_execute_model_async")
            ret = sdk.lyn_model_send_input_async(self.apu_stream_s, self.apu_model, apuinbuf, apuoutbuf, self.modelDict['batchsize'])
            error_check(ret != 0, "lyn_model_send_input_async")
            ret = sdk.lyn_model_recv_output_async(self.apu_stream_r, self.apu_model)
            error_check(ret != 0, "lyn_model_recv_output_async")
            apuinbuf = sdk.lyn_addr_seek(apuinbuf, self.modelDict['inputdatalen'] * self.modelDict['batchsize'])
            apuoutbuf = sdk.lyn_addr_seek(apuoutbuf, self.modelDict['outputdatalen'] * self.modelDict['batchsize'])
            if step == self.time_steps - 1:
                ret = sdk.lyn_record_event(self.apu_stream_r, self.mem_reset_event)
        # sdk.lyn_memcpy_async(self.apu_stream_r,self.hostOutbuf,self.apuOutbuf,self.modelDict['outputdatalen']*self.modelDict['batchsize']*self.time_steps,S2C)
        ret = sdk.lyn_stream_add_callback(self.apu_stream_r, get_result_callback, [self, currentInbuf, currentOutbuf])
        self.run_times += 1

        return ret

    def __call__(self, img):
        return self.run(img)

    def get_output(self):
        sdk.lyn_set_current_context(self.context)

        sdk.lyn_synchronize_stream(self.apu_stream_r)
        sdk.lyn_synchronize_stream(self.apu_stream_s)

        return self.output_list

    def _apu_unload(self):
        sdk.lyn_set_current_context(self.context)
        ret = 0
        for i in range(self.input_list_len):
            ret |= sdk.lyn_free(getattr(self, 'apuInbuf{}'.format(i)))
            ret |= sdk.lyn_free(getattr(self, 'apuOutbuf{}'.format(i)))
        sdk.c_free(self.hostOutbuf)
        ret |= sdk.lyn_unload_model(self.apu_model)
        ret |= sdk.lyn_destroy_stream(self.apu_stream_r)
        ret |= sdk.lyn_destroy_stream(self.apu_stream_s)
        ret |= sdk.lyn_destroy_context(self.context)
        error_check(ret != 0, "releae sdk resource failed")
    
    def __del__(self):
        self._apu_unload()

    def register_progressbar(self, bar):
        self.propressbar = bar

# for single time steps
class ApuRun_Single:
    def __init__(self, apu_device, apu_model_path):
        self.apu_device = apu_device

        assert os.path.exists(apu_model_path), "apu model path not exists"
        ret = 0

        self._sdk_initialize()

        self.count, ret = sdk.lyn_get_device_count()
        print("chip count = ", self.count)
        error_check(ret != 0, "lyn_get_device_count")
        assert self.count > self.apu_device, "device id must be less device count"

        self.apu_model, ret = sdk.lyn_load_model(apu_model_path)
        error_check(ret != 0, "lyn_load_model")
        self._model_parse()

    def _sdk_initialize(self):
        ret = 0

        self.context, ret = sdk.lyn_create_context(self.apu_device)
        error_check(ret != 0, "lyn_create_context")

        ret = sdk.lyn_set_current_context(self.context)
        error_check(ret != 0, "lyn_set_current_context")

        ret = sdk.lyn_register_error_handler(error_check_handler)
        error_check(ret != 0, "lyn_register_error_handler")
    
        self.apu_stream, ret = sdk.lyn_create_stream()
        error_check(ret != 0, "lyn_create_stream")
    
    def _model_parse(self):
        ret = 0
        self.modelDict = {}
        model_desc, ret = sdk.lyn_model_get_desc(self.apu_model)
        error_check(ret!=0, "lyn_model_get_desc")

        self.modelDict['batchsize'] = model_desc.inputTensorAttrArray[0].batchSize
        self.modelDict['inputnum'] = len(model_desc.inputTensorAttrArray)
        inputshapeList = []
        for i in range(self.modelDict['inputnum']):
            inputDims = len(model_desc.inputTensorAttrArray[i].dims)
            inputShape = []
            for j in range(inputDims):
                inputShape.append(model_desc.inputTensorAttrArray[i].dims[j])
            inputshapeList.append(inputShape)
        self.modelDict['inputshape'] = inputshapeList    
        self.modelDict['inputdatalen'] = model_desc.inputDataLen
        self.modelDict['inputdatatype'] = model_desc.inputTensorAttrArray[0].dtype
        self.modelDict['outputnum'] = len(model_desc.outputTensorAttrArray)
        outputshapeList = []
        for i in range(self.modelDict['outputnum']):
            outputDims = len(model_desc.outputTensorAttrArray[i].dims)
            outputShape = []
            for j in range(outputDims):
                outputShape.append(model_desc.outputTensorAttrArray[i].dims[j])
            outputshapeList.append(outputShape)
        self.modelDict['outputshape'] = outputshapeList
        self.modelDict['outputdatalen'] = model_desc.outputDataLen
        self.modelDict['outputdatatype'] = model_desc.outputTensorAttrArray[0].dtype
        # print(self.modelDict)
        print('######## model informations ########')
        for key, value in self.modelDict.items():
            print('{}: {}'.format(key,value))
        print('####################################')

        self.apuInbuf, ret = sdk.lyn_malloc(self.modelDict['inputdatalen'] * self.modelDict['batchsize'])
        error_check(ret != 0, "lyn_malloc")
        self.apuOutbuf,ret = sdk.lyn_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize']) 
        error_check(ret != 0, "lyn_malloc")
        self.hostOutbuf = sdk.c_malloc(self.modelDict['outputdatalen'] * self.modelDict['batchsize'])

    def run(self, img, time_steps):
        assert isinstance(img, np.ndarray)
        sdk.lyn_set_current_context(self.context)

        ret = 0
        img = img.astype(dtype_dict[self.modelDict['inputdatatype']][0])
        img_ptr, _ = sdk.lyn_numpy_contiguous_to_ptr(img)
        sdk.lyn_memcpy_async(self.apu_stream, self.apuInbuf, img_ptr, self.modelDict['inputdatalen'] * self.modelDict['batchsize'], C2S)
        
        if time_steps == 0:
            ret = sdk.lyn_model_reset_async(self.apu_stream, self.apu_model)
            error_check(ret!=0, "lyn_model_reset_async")

        ret = sdk.lyn_execute_model_async(self.apu_stream, self.apu_model, self.apuInbuf, self.apuOutbuf, self.modelDict['batchsize'])
        error_check(ret != 0, "lyn_execute_model_async")

        sdk.lyn_memcpy_async(self.apu_stream, self.hostOutbuf, self.apuOutbuf, self.modelDict['outputdatalen'] * self.modelDict['batchsize'], S2C)
        sdk.lyn_synchronize_stream(self.apu_stream)
       
        outputArrary = sdk.lyn_ptr_to_numpy(self.hostOutbuf, (self.modelDict['batchsize'], int(self.modelDict['outputdatalen'] / dtype_dict[self.modelDict['outputdatatype']][1]), ),\
            dtype_dict[self.modelDict['outputdatatype']][0])
        
        segment_lengths = []
        output = []
        for i in range(self.modelDict['outputnum']):
            segment_lengths.append(calculate_product(self.modelDict['outputshape'][i]))
        spilt_arr = split_array_by_lengths(outputArrary, segment_lengths, axis=-1)
        for i in range(self.modelDict['outputnum']):
            if len(self.modelDict['outputshape'][i]) == 1:
                output.append(spilt_arr[i].reshape(tuple(self.modelDict['outputshape'][i])))
            elif self.modelDict['batchsize'] == 1 and self.modelDict['batchsize'] != self.modelDict['outputshape'][i][0]:
                output.append(spilt_arr[i].reshape(tuple(self.modelDict['outputshape'][i])))
            else:
                output.append(spilt_arr[i].reshape((self.modelDict['batchsize'], ) + tuple(self.modelDict['outputshape'][i][1:])))

        return output

    def __call__(self, img, time_steps=None):
        if time_steps is not None:
            return self.run(img, time_steps)
        else:
            return self.run(img, 1)
        
    def apu_unload(self):
        ret = 0        
        sdk.lyn_set_current_context(self.context)
        ret |= sdk.lyn_free(self.apuInbuf)
        ret |= sdk.lyn_free(self.apuOutbuf)
        sdk.c_free(self.hostOutbuf)
        ret |= sdk.lyn_unload_model(self.apu_model)
        ret |= sdk.lyn_destroy_stream(self.apu_stream)        
        ret |= sdk.lyn_destroy_context(self.context)
        error_check(ret != 0, "releae sdk resource failed")
    
    # def __del__(self):
    #     self.apu_unload()
        

# for multi models
def get_result_callback2(params):
    self = params[0]
    model_id = params[1]
    time_steps = params[2]

    sdk.lyn_memcpy(self.hostOutbufs[model_id], self.apuOutbufs[model_id], self.modelDicts[model_id]['outputdatalen'] * self.modelDicts[model_id]['batchsize'], S2C)
    
    outputArrary = sdk.lyn_ptr_to_numpy(self.hostOutbufs[model_id], (self.modelDicts[model_id]['batchsize'], int(self.modelDicts[model_id]['outputdatalen'] / dtype_dict[self.modelDicts[model_id]['outputdatatype']][1]), ),\
        dtype_dict[self.modelDicts[model_id]['outputdatatype']][0])
    
    segment_lengths = []
    output = []
    for i in range(self.modelDicts[model_id]['outputnum']):
        segment_lengths.append(calculate_product(self.modelDicts[model_id]['outputshape'][i]))
    spilt_arr = split_array_by_lengths(outputArrary, segment_lengths, axis=-1)
    for i in range(self.modelDicts[model_id]['outputnum']):
        if len(self.modelDicts[model_id]['outputshape'][i]) == 1:
            output.append(spilt_arr[i].reshape(tuple(self.modelDicts[model_id]['outputshape'][i])))
        elif self.modelDicts[model_id]['batchsize'] == 1 and self.modelDicts[model_id]['batchsize'] != self.modelDicts[model_id]['outputshape'][i][0]:
            output.append(spilt_arr[i].reshape(tuple(self.modelDicts[model_id]['outputshape'][i])))
        else:
            output.append(spilt_arr[i].reshape((self.modelDicts[model_id]['batchsize'], ) + tuple(self.modelDicts[model_id]['outputshape'][i][1:])))
    
    self.outputs[model_id].append(output)

    if self.propressbars[model_id] is not None and time_steps == self.max_timesteps-1:
        self.propressbars[model_id].update()

    self.keep_alive.pop(0)

    return 0

class ApuRun_Multi_Models:
    def __init__(self, apu_device, apu_model_paths=[]):
        self.apu_device = apu_device

        assert apu_model_paths != [], "apu model path not exists"
        ret = 0

        self._sdk_initialize()

        self.count, ret = sdk.lyn_get_device_count()
        print("chip count = ", self.count)
        error_check(ret != 0, "lyn_get_device_count")
        assert self.count > self.apu_device, "device id must be less device count"

        self.apu_models = []
        for path in apu_model_paths:
            apu_model, ret = sdk.lyn_load_model(path)
            error_check(ret != 0, "lyn_load_model")
            self.apu_models.append(apu_model)
        self._model_parse()

        self.keep_alive = []
        self.propressbars = [None for _ in range(len(apu_model_paths))]
        self.max_timesteps = 1
        self.outputs = [[] for _ in range(len(apu_model_paths))]

    def _sdk_initialize(self):
        ret = 0

        self.context, ret = sdk.lyn_create_context(self.apu_device)
        error_check(ret != 0, "lyn_create_context")

        ret = sdk.lyn_set_current_context(self.context)
        error_check(ret != 0, "lyn_set_current_context")

        ret = sdk.lyn_register_error_handler(error_check_handler)
        error_check(ret != 0, "lyn_register_error_handler")
    
        self.apu_stream, ret = sdk.lyn_create_stream()
        error_check(ret != 0, "lyn_create_stream")
    
    def _model_parse(self):
        self.modelDicts = []
        self.apuInbufs = []
        self.apuOutbufs = []
        self.hostOutbufs = []

        for apu_model in self.apu_models:
            ret = 0
            modelDict = {}
            model_desc, ret = sdk.lyn_model_get_desc(apu_model)
            error_check(ret!=0, "lyn_model_get_desc")

            modelDict['batchsize'] = model_desc.inputTensorAttrArray[0].batchSize
            modelDict['inputnum'] = len(model_desc.inputTensorAttrArray)
            inputshapeList = []
            for i in range(modelDict['inputnum']):
                inputDims = len(model_desc.inputTensorAttrArray[i].dims)
                inputShape = []
                for j in range(inputDims):
                    inputShape.append(model_desc.inputTensorAttrArray[i].dims[j])
                inputshapeList.append(inputShape)
            modelDict['inputshape'] = inputshapeList    
            modelDict['inputdatalen'] = model_desc.inputDataLen
            modelDict['inputdatatype'] = model_desc.inputTensorAttrArray[0].dtype
            modelDict['outputnum'] = len(model_desc.outputTensorAttrArray)
            outputshapeList = []
            for i in range(modelDict['outputnum']):
                outputDims = len(model_desc.outputTensorAttrArray[i].dims)
                outputShape = []
                for j in range(outputDims):
                    outputShape.append(model_desc.outputTensorAttrArray[i].dims[j])
                outputshapeList.append(outputShape)
            modelDict['outputshape'] = outputshapeList
            modelDict['outputdatalen'] = model_desc.outputDataLen
            modelDict['outputdatatype'] = model_desc.outputTensorAttrArray[0].dtype
            # print(self.modelDict)
            print('######## model informations ########')
            for key, value in modelDict.items():
                print('{}: {}'.format(key,value))
            print('####################################')

            apuInbuf, ret = sdk.lyn_malloc(modelDict['inputdatalen'] * modelDict['batchsize'])
            error_check(ret != 0, "lyn_malloc")
            apuOutbuf,ret = sdk.lyn_malloc(modelDict['outputdatalen'] * modelDict['batchsize']) 
            error_check(ret != 0, "lyn_malloc")
            hostOutbuf = sdk.c_malloc(modelDict['outputdatalen'] * modelDict['batchsize'])

            self.modelDicts.append(modelDict)
            self.apuInbufs.append(apuInbuf)
            self.apuOutbufs.append(apuOutbuf)
            self.hostOutbufs.append(hostOutbuf)

    def run(self, img, time_steps=1, model_id=0):
        assert isinstance(img, np.ndarray)
        sdk.lyn_set_current_context(self.context)

        ret = 0
        img = img.astype(dtype_dict[self.modelDicts[model_id]['inputdatatype']][0])
        img_ptr, _ = sdk.lyn_numpy_contiguous_to_ptr(img)
        # self.keep_alive.append(img_ptr)
        sdk.lyn_memcpy_async(self.apu_stream, self.apuInbufs[model_id], img_ptr, self.modelDicts[model_id]['inputdatalen'] * self.modelDicts[model_id]['batchsize'], C2S)

        if time_steps == 0:
            ret = sdk.lyn_model_reset_async(self.apu_stream, self.apu_models[model_id])
            error_check(ret!=0, "lyn_model_reset_async")
        ret = sdk.lyn_execute_model_async(self.apu_stream, self.apu_models[model_id], self.apuInbufs[model_id], self.apuOutbufs[model_id], self.modelDicts[model_id]['batchsize'])
        error_check(ret != 0, "lyn_execute_model_async")
        
        sdk.lyn_memcpy_async(self.apu_stream, self.hostOutbufs[model_id], self.apuOutbufs[model_id], self.modelDicts[model_id]['outputdatalen'] * self.modelDicts[model_id]['batchsize'], S2C)
        sdk.lyn_synchronize_stream(self.apu_stream)
        
        outputArrary = sdk.lyn_ptr_to_numpy(self.hostOutbufs[model_id], (self.modelDicts[model_id]['batchsize'], int(self.modelDicts[model_id]['outputdatalen'] / dtype_dict[self.modelDicts[model_id]['outputdatatype']][1]), ),\
            dtype_dict[self.modelDicts[model_id]['outputdatatype']][0])
    
        segment_lengths = []
        output = []
        for i in range(self.modelDicts[model_id]['outputnum']):
            segment_lengths.append(calculate_product(self.modelDicts[model_id]['outputshape'][i]))
        spilt_arr = split_array_by_lengths(outputArrary, segment_lengths, axis=-1)
        for i in range(self.modelDicts[model_id]['outputnum']):
            if len(self.modelDicts[model_id]['outputshape'][i]) == 1:
                output.append(spilt_arr[i].reshape(tuple(self.modelDicts[model_id]['outputshape'][i])))
            elif self.modelDicts[model_id]['batchsize'] == 1 and self.modelDicts[model_id]['batchsize'] != self.modelDicts[model_id]['outputshape'][i][0]:
                output.append(spilt_arr[i].reshape(tuple(self.modelDicts[model_id]['outputshape'][i])))
            else:
                output.append(spilt_arr[i].reshape((self.modelDicts[model_id]['batchsize'], ) + tuple(self.modelDicts[model_id]['outputshape'][i][1:])))

        # ret = sdk.lyn_stream_add_callback(self.apu_stream, get_result_callback2, [self, model_id, time_steps])
        # error_check(ret != 0, "lyn_stream_add_callback")
        return output
    
    # def wait_for_run(self):
    #     sdk.lyn_set_current_context(self.context)
    #     sdk.lyn_synchronize_stream(self.apu_stream)

    # def get_outputs(self):
    #     return self.outputs

    def apu_unload(self):
        ret = 0        
        sdk.lyn_set_current_context(self.context)
        for hostOutbuf in self.hostOutbufs:
            sdk.c_free(hostOutbuf)
        for apuInbuf in self.apuInbufs:
            ret |= sdk.lyn_free(apuInbuf)
        for apuOutbuf in self.apuOutbufs:
            ret |= sdk.lyn_free(apuOutbuf)
        for apu_model in self.apu_models:
            ret |= sdk.lyn_unload_model(apu_model)
        ret |= sdk.lyn_destroy_stream(self.apu_stream)   
        ret |= sdk.lyn_destroy_context(self.context)
        error_check(ret != 0, "releae sdk resource failed")

    
    def register_progressbar(self, bar, model_id, max_timesteps):
        self.propressbars[model_id] = bar
        self.max_timesteps = max_timesteps
    
