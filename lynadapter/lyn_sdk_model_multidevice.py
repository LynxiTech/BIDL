# Copyright (c) Lynxi Technologies Co., Ltd. All rights reserved.
# Copyright (c) China Nanhu Academy of Electronics and Information Technology. All rights reserved.


# import sys
import pylynchipsdk as sdk
import numpy as np
import json
import time

SDK_DYTPE = {
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

def callback(params):
    print('callback: {}'.format(params[0]))
    return 0

def stream_error(stream, errorMsg, params):
    print("******* streamID : ", stream)
    print("******* errorCode : {}".format(errorMsg.errCode))

time_record = False
callback_trace = False
via_p2p = False
class ApuRun:
    h_mem_addr_arr = None
    device_arr = None
    d_output_addr_arr = None
    d_input_addr_arr = None
    infer_event_arr = None
    output_event_arr = None
    mem_event_arr = None
    if via_p2p == False:
        h_input_addr_arr = None

    def __init__(self, group_id, id, devices, model_path, time_steps):
        self.group_id = group_id
        self.id = id
        self.devices = devices
        self.device_id = devices[self.group_id][self.id]
        self.dp_num = len(self.devices)
        self.mp_num = len(self.devices[self.group_id])
        self.device_count = self.dp_num * self.mp_num
        self.model_path = model_path
        self.time_steps = time_steps
        self.first_run = True
        self.run_num = 0
        self.infer_time = 0
        self.i_cpy_time = 0
        self.iter_infer_time = 0
        # self.infer_count = 0
        try:
            # self.log_init()
            if self.group_id == 0 and self.id == 0:
                self.global_init()
            self.context, self.stream, self.mem_sent_event, self.dp_infer_completed_event, self.mp_infer_completed_event,\
                self.output_sent_event, self.infer_start_event, self.infer_end_event = self.sdk_init(self.device_id)
            self.model, self.model_desc, self.total_output_data_len, self.batch_size, self.host_out_ptr, \
            self.dev_out_ptr, self.dev_ptr, self.total_input_data_len, self.mem_phyaddr, self.mem_size, self.host_mem_ptr, self.host_input_ptr = \
                self.load_model(model_path)
            self._base_i_addr = self.dev_ptr
            self._base_o_addr = self.dev_out_ptr
        except:
            self.release()
            raise Exception('__init__ failed')

    def global_init(self):
        ApuRun.device_arr = np.array(self.devices)
        ApuRun.h_mem_addr_arr = np.empty((self.dp_num, self.mp_num), dtype=object)
        ApuRun.d_output_addr_arr = np.empty((self.dp_num, self.mp_num), dtype=object)
        ApuRun.d_input_addr_arr = np.empty((self.dp_num, self.mp_num), dtype=object)
        ApuRun.infer_event_arr = np.empty((self.dp_num, self.mp_num, 2), dtype=object) # 0: dp, 1: mp
        ApuRun.output_event_arr = np.empty((self.dp_num, self.mp_num), dtype=object)
        ApuRun.mem_event_arr = np.empty((self.dp_num, self.mp_num), dtype=object)
        ApuRun.h_input_addr_arr = np.empty((self.dp_num, self.mp_num), dtype=object)

    def log_init(self):
        log_attr = sdk.lyn_client_log_attr_t()
        log_attr.level = sdk.lyn_log_level_t.ERROR
        log_attr.is_save_file = False
        ret = sdk.lyn_log_init_client(log_attr)
        assert ret == 0, '``lyn_log_init_client`` fail'

        log_attr_server = sdk.lyn_server_log_attr_t()
        log_attr_server.device_id = 0
        log_attr_server.level = sdk.lyn_log_level_t.ERROR
        log_attr_server.is_save_file = False
        ret = sdk.lyn_log_init_server(log_attr_server)
        assert ret == 0, '``lyn_log_init_server`` fail'

    # @staticmethod
    def sdk_init(self, device_id):
        context, ret = sdk.lyn_create_context(device_id)
        assert ret == 0, '``lyn_create_context`` fail'

        stream, ret = sdk.lyn_create_stream()
        assert ret == 0, '``lyn_create_stream`` fail'

        ret = sdk.lyn_register_error_handler(stream_error)
        assert ret == 0, '``lyn_register_error_handler`` fail'

        dp_infer_completed_event, ret = sdk.lyn_create_event()
        assert ret == 0, '``lyn_create_event`` fail'
        ApuRun.infer_event_arr[self.group_id][self.id][0] = dp_infer_completed_event

        mp_infer_completed_event, ret = sdk.lyn_create_event()
        assert ret == 0, '``lyn_create_event`` fail'
        ApuRun.infer_event_arr[self.group_id][self.id][1] = mp_infer_completed_event

        mem_sent_event, ret = sdk.lyn_create_event()
        assert ret == 0, '``lyn_create_event`` fail'
        ApuRun.mem_event_arr[self.group_id][self.id] = mem_sent_event

        output_sent_event, ret = sdk.lyn_create_event()
        assert ret == 0, '``lyn_create_event`` fail'
        ApuRun.output_event_arr[self.group_id][self.id] = output_sent_event

        infer_start_event, ret = sdk.lyn_create_event()
        assert ret == 0, '``lyn_create_event`` fail'

        infer_end_event, ret = sdk.lyn_create_event()
        assert ret == 0, '``lyn_create_event`` fail'

        return context, stream, mem_sent_event, dp_infer_completed_event, mp_infer_completed_event, \
            output_sent_event, infer_start_event, infer_end_event

    def logger(self, enable, msg):
        if enable:
            sdk.lyn_stream_add_async_callback(self.stream, callback, msg)

    def load_json(self, path):
        path = path
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        # print(data)
        fp.close()
        return data

    def load_model(self, model_path):
        ret = sdk.lyn_set_current_context(self.context)

        model, ret = sdk.lyn_load_model(model_path)
        assert ret == 0, '``lyn_load_model`` fail'

        model_desc, ret = sdk.lyn_model_get_desc(model)
        assert ret == 0, '``lyn_model_get_desc`` fail'

        total_input_data_len = model_desc.inputDataLen
        # print("model's input tensor length = ", total_input_data_len)         
        dev_ptr, ret = sdk.lyn_malloc(total_input_data_len * self.time_steps)
        assert ret == 0, "``lyn_malloc`` fail"
        ApuRun.d_input_addr_arr[self.group_id][self.id] = dev_ptr
        if via_p2p == False:
            host_input_ptr = sdk.c_malloc(total_input_data_len * self.time_steps)
            assert host_input_ptr, "``c_malloc`` fail"
            ApuRun.h_input_addr_arr[self.group_id][self.id] = host_input_ptr
        else:
            host_input_ptr = None

        total_output_data_len = model_desc.outputDataLen
        # print("model's output tensor length = ", total_output_data_len)        
        batch_size = model_desc.inputTensorAttrArray[0].batchSize

        host_out_ptr = sdk.c_malloc(total_output_data_len)
        assert host_out_ptr, '``c_malloc`` fail'

        dev_out_ptr, ret = sdk.lyn_malloc(total_output_data_len * self.time_steps)
        assert ret == 0, "``lyn_malloc`` fail"
        ApuRun.d_output_addr_arr[self.group_id][self.id] = dev_out_ptr

        apu_json = self.model_path + '/apu_0/apu_x/apu.json'
        apu_json = self.load_json(apu_json)
        # print('mem_addr = 0x%x'%apu_json['apu_x']['userload']['addr'])
        # print('mem_size = ', apu_json['apu_x']['userload']['size'])

        mem_phyaddr = apu_json['apu_x']['userload']["info"][0]['addr']
        mem_size = apu_json['apu_x']['userload']["info"][0]['size']

        host_mem_ptr = sdk.c_malloc(mem_size)
        assert host_out_ptr, '``c_malloc`` fail'
        ApuRun.h_mem_addr_arr[self.group_id][self.id] = host_mem_ptr

        return model, model_desc, total_output_data_len, batch_size, host_out_ptr, dev_out_ptr, dev_ptr, \
            total_input_data_len, mem_phyaddr, mem_size, host_mem_ptr, host_input_ptr

    def load_data(self, data):
        sdk.lyn_set_current_context(self.context)
        # assert data.dtype == SDK_DYTPE[self.model_desc.inputTensorAttrArray[0].dtype][0], \
        #     "dtype of input numpy dtype is mismatch, dtype of input is {}".format(SDK_DYTPE[self.model_desc.inputTensorAttrArray[0].dtype][0])
        data = data.astype(SDK_DYTPE[self.model_desc.inputTensorAttrArray[0].dtype][0])
        host_ptr = sdk.lyn_numpy_to_ptr(data)
        ret = sdk.lyn_memcpy(self._base_i_addr, host_ptr, self.total_input_data_len * self.time_steps,
                       sdk.lyn_memcpy_dir_t.ClientToServer)
        assert ret == 0, '``lyn_memcpy`` fail'
        
    def run(self, data=None):
        # print('[group:{},id:{}]: run start {} ...'.format(self.group_id, self.id, self.run_num))
        sdk.lyn_set_current_context(self.context)
        ret = sdk.lyn_model_reset_async(self.stream, self.model)
        assert ret == 0, '``lyn_model_reset_async`` fail'
        if isinstance(data, np.ndarray):
            # self.load_data(data)
            # assert data.dtype == SDK_DYTPE[self.model_desc.inputTensorAttrArray[0].dtype][0], \
            #     "dtype of input numpy dtype is mismatch, dtype of input is {}".format(SDK_DYTPE[self.model_desc.inputTensorAttrArray[0].dtype][0])
            data = data.astype(SDK_DYTPE[self.model_desc.inputTensorAttrArray[0].dtype][0])
            host_ptr = sdk.lyn_numpy_to_ptr(data)
            ret = sdk.lyn_memcpy_async(self.stream, self._base_i_addr, host_ptr, self.total_input_data_len * self.time_steps,
                        sdk.lyn_memcpy_dir_t.ClientToServer)
            assert ret == 0, '``lyn_memcpy_async`` fail'
        else:
            self.logger(callback_trace, ['[group:{},id:{}]: wait output_sent_event from [group:{},id:{}]'\
                    .format(self.group_id, self.id, self.group_id, self.id-1)])
            ret = sdk.lyn_stream_wait_event(self.stream, ApuRun.output_event_arr[self.group_id][self.id-1])
            assert ret == 0, '``lyn_stream_wait_event`` fail'
            self.logger(callback_trace, ['[group:{},id:{}]: receive output_sent_event from [group:{},id:{}]'\
                    .format(self.group_id, self.id, self.group_id, self.id-1)])
            if via_p2p == False:
                ret = sdk.lyn_memcpy_async(self.stream, self._base_i_addr, self.host_input_ptr, self.total_input_data_len*self.time_steps,
                                sdk.lyn_memcpy_dir_t.ClientToServer)
                assert ret == 0, '``lyn_memcpy_async`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: write input data to server'\
                        .format(self.group_id, self.id)])

        if self.group_id != 0:
            self.logger(callback_trace, ['[group:{},id:{}]: wait mem_sent_event from [group:{},id:{}]'\
                    .format(self.group_id, self.id, self.group_id-1, self.id)])
            ret = sdk.lyn_stream_wait_event(self.stream, ApuRun.mem_event_arr[self.group_id -1][self.id])
            assert ret == 0, '``lyn_stream_wait_event`` fail'
            self.logger(callback_trace, ['[group:{},id:{}]: receive mem_sent_event from [group:{},id:{}]'\
                    .format(self.group_id, self.id, self.group_id-1, self.id)])
            ret = sdk.lyn_memcpy_async_ex(self.stream, self.mem_phyaddr, self.host_mem_ptr, self.mem_size,
                                          sdk.lyn_memcpy_dir_t.ClientToServer)                            
            assert ret == 0, '``lyn_memcpy_async_ex`` fail'
            self.logger(callback_trace, ['[group:{},id:{}]: write mem from client to server'\
                    .format(self.group_id, self.id)])
  
        if time_record:
            ret = sdk.lyn_record_event(self.stream, self.infer_start_event)
            assert ret == 0, '``lyn_record_event`` fail'
        self.dev_ptr = self._base_i_addr
        self.dev_out_ptr = self._base_o_addr
        for run_time in range(self.time_steps):
            ret = sdk.lyn_execute_model_async(self.stream, self.model, self.dev_ptr, self.dev_out_ptr,
                                              self.batch_size)
            assert ret == 0, '``lyn_execute_model_async`` fail'
            self.dev_ptr = sdk.lyn_addr_seek(self.dev_ptr, self.total_input_data_len)
            self.dev_out_ptr = sdk.lyn_addr_seek(self.dev_out_ptr, self.total_output_data_len)
            assert ret == 0, '``lyn_addr_seek`` fail'
        self.logger(callback_trace, ['[group:{},id:{}]: inference completed'\
                .format(self.group_id, self.id)])
        if time_record:
            ret = sdk.lyn_record_event(self.stream, self.infer_end_event)
            assert ret == 0, '``lyn_record_event`` fail'

        if self.group_id != 0 or self.id != 0:
            if self.group_id == 0:
                ret = sdk.lyn_record_event(self.stream, self.mp_infer_completed_event)
                assert ret == 0, '``lyn_record_event`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: send mp_infer_completed_event'\
                        .format(self.group_id, self.id)])
            elif self.id == 0:
                ret = sdk.lyn_record_event(self.stream, self.dp_infer_completed_event)
                assert ret == 0, '``lyn_record_event`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: send dp_infer_completed_event'\
                        .format(self.group_id, self.id)])
            else:
                ret = sdk.lyn_record_event(self.stream, self.dp_infer_completed_event)
                assert ret == 0, '``lyn_record_event`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: send dp_infer_completed_event'\
                        .format(self.group_id, self.id)])
                ret = sdk.lyn_record_event(self.stream, self.mp_infer_completed_event)
                assert ret == 0, '``lyn_record_event`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: send mp_infer_completed_event'\
                        .format(self.group_id, self.id)])

        if self.first_run == False:
            if self.group_id != self.dp_num - 1 or self.id != self.mp_num - 1:
                if self.id == self.mp_num - 1:
                    self.logger(callback_trace, ['[group:{},id:{}]: wait dp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id+1, self.id)])
                    ret = sdk.lyn_stream_wait_event(self.stream, ApuRun.infer_event_arr[self.group_id+1][self.id][0])
                    assert ret == 0, '``lyn_stream_wait_event`` fail'
                    self.logger(callback_trace, ['[group:{},id:{}]: receive dp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id+1, self.id)])
                elif self.group_id == self.dp_num - 1:
                    self.logger(callback_trace, ['[group:{},id:{}]: wait mp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id, self.id+1)])
                    ret = sdk.lyn_stream_wait_event(self.stream, ApuRun.infer_event_arr[self.group_id][self.id+1][1])
                    assert ret == 0, '``lyn_stream_wait_event`` fail'
                    self.logger(callback_trace, ['[group:{},id:{}]: receive mp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id, self.id+1)])
                else:
                    self.logger(callback_trace, ['[group:{},id:{}]: wait dp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id+1, self.id)])
                    ret = sdk.lyn_stream_wait_event(self.stream, ApuRun.infer_event_arr[self.group_id+1][self.id][0])
                    assert ret == 0, '``lyn_stream_wait_event`` fail'
                    self.logger(callback_trace, ['[group:{},id:{}]: receive dp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id+1, self.id)])
                    self.logger(callback_trace, ['[group:{},id:{}]: wait mp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id, self.id+1)])
                    ret = sdk.lyn_stream_wait_event(self.stream, ApuRun.infer_event_arr[self.group_id][self.id+1][1])
                    assert ret == 0, '``lyn_stream_wait_event`` fail'
                    self.logger(callback_trace, ['[group:{},id:{}]: receive mp_infer_completed_event from [group:{},id:{}]'\
                            .format(self.group_id, self.id, self.group_id, self.id+1)])

        if self.id != self.mp_num - 1:
            if via_p2p:
                ret = sdk.lyn_memcpy_async(self.stream, ApuRun.d_input_addr_arr[self.group_id][self.id+1], self._base_o_addr, self.total_output_data_len*self.time_steps,
                                sdk.lyn_memcpy_dir_t.ServerToServer)
                assert ret == 0, '``lyn_memcpy_async`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: write output to [group:{},id:{}]'\
                        .format(self.group_id, self.id, self.group_id, self.id+1)])
            else:
                ret = sdk.lyn_memcpy_async(self.stream, ApuRun.h_input_addr_arr[self.group_id][self.id+1], self._base_o_addr, self.total_output_data_len*self.time_steps,
                                sdk.lyn_memcpy_dir_t.ServerToClient)
                assert ret == 0, '``lyn_memcpy_async`` fail'
                self.logger(callback_trace, ['[group:{},id:{}]: write output to [group:{},id:{}] host input address'\
                        .format(self.group_id, self.id, self.group_id, self.id+1)])
            ret = sdk.lyn_record_event(self.stream, self.output_sent_event)
            assert ret == 0, '``lyn_record_event`` fail'
            self.logger(callback_trace, ['[group:{},id:{}]: send output_sent_event'\
                    .format(self.group_id, self.id)] )
            
        if self.group_id != self.dp_num - 1:
            ret = sdk.lyn_memcpy_async_ex(self.stream, self.mem_phyaddr, ApuRun.h_mem_addr_arr[self.group_id+1][self.id],
                                          self.mem_size, sdk.lyn_memcpy_dir_t.ServerToClient)
            assert ret == 0, '``lyn_memcpy_async_ex`` fail'
            self.logger(callback_trace, ['[group:{},id:{}]: read mem from server to client'\
                    .format(self.group_id, self.id)])
            ret = sdk.lyn_record_event(self.stream, self.mem_sent_event)
            assert ret == 0, '``lyn_record_event`` fail'
            self.logger(callback_trace, ['[group:{},id:{}]: send mem_sent_event'\
                    .format(self.group_id, self.id)])       

        ret = sdk.lyn_synchronize_stream(self.stream)
        assert ret == 0, '``lyn_synchronize_stream`` fail'

        if time_record:
            self.iter_infer_time, ret = sdk.lyn_event_elapsed_time(self.infer_start_event, self.infer_end_event)
            # print('[group:{},id:{}]: infer time = {}'.format(self.group_id, self.id, self.iter_infer_time))
            self.infer_time += self.iter_infer_time
        self.first_run = False
        # self.run_num += 1

    def get_output(self):
        sdk.lyn_set_current_context(self.context)
        self.dev_out_ptr = sdk.lyn_addr_seek(self._base_o_addr, self.total_output_data_len*(self.time_steps-1))
        ret = sdk.lyn_memcpy(self.host_out_ptr, self.dev_out_ptr, self.total_output_data_len,
                             sdk.lyn_memcpy_dir_t.ServerToClient)
        assert ret == 0, '``lyn_memcpy`` fail'
        type_size = SDK_DYTPE[self.model_desc.outputTensorAttrArray[0].dtype][1]
        output = sdk.lyn_ptr_to_numpy(self.host_out_ptr, (1, int(self.total_output_data_len / type_size)),
                                      self.model_desc.outputTensorAttrArray[0].dtype)
        return output

    def release(self):
        sdk.lyn_set_current_context(self.context)
        if hasattr(self, 'host_out_ptr'):
            sdk.c_free(self.host_out_ptr)
        if hasattr(self, 'host_input_ptr') and self.host_input_ptr != None:
            sdk.c_free(self.host_input_ptr)
        if hasattr(self, 'host_mem_ptr'): 
            sdk.c_free(self.host_mem_ptr)
        if hasattr(self, '_base_o_addr'):
            sdk.lyn_free(self._base_o_addr)
        if hasattr(self, '_base_i_addr'):
            sdk.lyn_free(self._base_i_addr)
        if hasattr(self, 'model'):
            sdk.lyn_unload_model(self.model)
        if hasattr(self, 'output_sent_event'):
            sdk.lyn_destroy_event(self.output_sent_event)
        if hasattr(self, 'mem_sent_event'):
            sdk.lyn_destroy_event(self.mem_sent_event)
        if hasattr(self, 'mp_infer_completed_event'):
            sdk.lyn_destroy_event(self.mp_infer_completed_event)
        if hasattr(self, 'dp_infer_completed_event'):
            sdk.lyn_destroy_event(self.dp_infer_completed_event)
        if hasattr(self, 'stream'):
            sdk.lyn_destroy_stream(self.stream)
        if hasattr(self, 'context'):
            sdk.lyn_destroy_context(self.context)
