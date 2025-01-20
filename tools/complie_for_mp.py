import os
import argparse
import torch
import sys; sys.path.append('../')
from lynadapter import model_operate
from utils.config import file2dict,pretty_text
import lynadapter.custom_op_in_lyn.custom_op_my_lif
from torch import ops
from layers import lif
from layers import lifplus
from applications.classification import *
from backbones import *
from datasets import *
from utils import globals
import re
globals._init()

lif.spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)
lifplus.spike_func = lambda _: torch.gt(_, 0.).to(_.dtype)

def load_library():
    # load libcustom_op.so
    library_path = "./lynadapter/custom_op_in_pytorch/build/libcustom_ops.so"
    ops.load_library(library_path)
def map_keys(state_dict,config):

    mapped_state_dict = {}
    
    if "it-" in config:
        patterns = [
            (re.compile(r'^conv.(\d+)\.(.+)$'), lambda m: f'conv.module.{m.group(1)}.{m.group(2)}'),
            (re.compile(r'^(.*?)\.conv(\d+)\.(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.conv{m.group(2)}.module.{m.group(3)}.{m.group(4)}'),
            (re.compile(r'^(.*?)\.downsample.(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.downsample.module.{m.group(2)}.{m.group(3)}'),
            #(re.compile(r'^([^.]+)\.lif.(.+)$'), lambda m: f'{m.group(1)}.lif.lif.{m.group(2)}'),
            #(re.compile(r'^(.*?)\.lif(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.lif{m.group(2)}.lif.{m.group(3)}')
        ]
    elif "cifar10" in config:
        return state_dict
    else:
        patterns = [
            (re.compile(r'^([^.]+)\.lif.(.+)$'), lambda m: f'{m.group(1)}.lif.lif.{m.group(2)}'),
            (re.compile(r'^(.*?)\.lif(\d+)\.(.+)$'), lambda m: f'{m.group(1)}.lif{m.group(2)}.lif.{m.group(3)}')
        ]
    
    
    for key, value in state_dict.items():
        for pattern, mapper in patterns:
            match = pattern.match(key)
            if match:
                mapped_key = mapper(match)
                mapped_state_dict[mapped_key] = value
                break  
        else:
            
            mapped_state_dict[key] = value
    
    return mapped_state_dict
def main():
    os.chdir('../')
    load_library()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='./applications/classification/dvs/cifar10-dvs/resnetlif50-cifar10dvs/resnetlif50-itout-b8x1-cifar10dvs_mp.py', help="test config file path")
    parser.add_argument("--checkpoint", default='./weight_files/classification/resnetlif18-t16/lif/latest.pth', help="checkpoint file name")
    args = parser.parse_args()
    application_abbr = { "luna16cls": "three_D/luna16cls/clif3fc3lc/",
                         "clif3fc3dm_itout-b16x1-dvsmnist":"dvs/dvs_mnist/clif3fc3dm/",
                         "clifplus3fc3dm_itout-b16x1-dvsmnist":"dvs/dvs_mnist/clif3fc3dm/",
                         "clif5fc2dm_itout-b16x1-dvsmnist":"dvs/dvs_mnist/clif5fc2dm/",
                         "clif5fc2dm_it-b16x1-dvsmnist":"dvs/dvs_mnist/clif5fc2dm/",
                         "clif3flif2dg_itout-b16x1-dvsgesture":"dvs/dvs_gesture/clif3flif2dg/",
                         "clifplus3flifplus2dg_itout-b16x1-dvsgesture":"dvs/dvs_gesture/clif3flif2dg/",
                         "clif7fc1dg_itout-b16x1-dvsgesture":"dvs/dvs_gesture/clif7fc1dg/",
                         "clif7fc1dg_it-b16x1-dvsgesture":"dvs/dvs_gesture/clif7fc1dg/",
                         "clif5fc2cd_itout-b64x1-cifar10dvs": "dvs/cifar10dvs/vgg_cifar10dvs/clif5fc2cd/",
                         "clif5fc2cd_it-b64x1-cifar10dvs":"dvs/cifar10dvs/vgg_cifar10dvs/clif5fc2cd/",
                         "clif7fc1cd_itout-b64x1-cifar10dvs":"dvs/cifar10dvs/vgg_cifar10dvs/clif7fc1cd/",
                         "clif7fc1cd_it-b64x1-cifar10dvs":"dvs/cifar10dvs/vgg_cifar10dvs/clif7fc1cd/",
                         "clifplus5fc2cd_itout-b64x1-cifar10dvs": "dvs/cifar10dvs/vgg_cifar10dvs/clif5fc2cd/",
                         "attentionVGG-it-b64x1-cifar10dvs": "dvs/cifar10dvs/vgg_cifar10dvs/attentionVGG/",
                         "resnetlif18-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif18_cifar10dvs/",
                         "resnetlif50-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif50_cifar10dvs/",
                         "resnetlif50-itout-b8x1-cifar10dvs_mp": "dvs/cifar10dvs/resnetlif50_cifar10dvs/",
                         "resnetlif50-lite-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif50_lite_cifar10dvs/",
                         "resnetlif18-lite-itout-b8x1-cifar10dvs": "dvs/cifar10dvs/resnetlif18_lite_cifar10dvs/",
                         "resnetlif50-it-b16x1-cifar10dvs": "dvs/cifar10dvs/resnetlif50_cifar10dvs/",
                         "rgbgesture": "videodiff/rgbgesture/clif3flif2rg/",
                         "resnetlif18-itout-b20x4-16-jester": "video/jester/resnetlif18_t16/",
                         "resnetlif18-itout-b16x4-8-jester": "video/jester/resnetlif18_t8/",
                         "resnetlif18-lite-itout-b20x4-16-jester": "video/jester/resnetlif18_lite_t16/",
                         "esimagenet": "dvs/esimagenet/resnetlif18ES/",
                         "imdb":"text/imdb/fasttextIM/",
                         "mnist":"spikegen/clif3fc3mn/"}
    data_name = args.config
    args.checkpoint='./weight_files/classification/resnetlif50_cifar10dvs/lif/'+args.checkpoint
    print('args.checkpoint = ', args.checkpoint)
    print('args.config = ', args.config)
    args.config = "./applications/classification/" + application_abbr[data_name] + args.config + '.py' 
    #cfg = mmcv.Config.fromfile(args.config)
    filename = os.path.basename(args.config)
    cfg = file2dict(args.config)   
    models_compile_inputshape = cfg['models_compile_inputshape']
    print('models_compile_inputshape = ', models_compile_inputshape)

    datasets_abbr = {"dvsmnist": "Dm",
                     "luna16cls": "Lc",
                     "dvsgesture": "Dg",
                     "cifar10dvs": "Cd",
                     "rgbgesture": "Rg",
                     "jester": "Jt"}
    model_num = len(models_compile_inputshape)
    for i in range(model_num):
        name = str(cfg['model_'+str(i)])
        globals.set_value('ON_APU', True)
        globals.set_value('FIT', True)
        if 'soma_params' in eval(name)['backbone'] and eval(name)['backbone']['soma_params'] == 'channel_share':
            globals.set_value('FIT', False)
        backbone_dict = cfg["model_"+str(i)]["backbone"]
        model_backbone_type =  backbone_dict.pop('type')
        model = eval(model_backbone_type)(**backbone_dict)
        backbone_dict['type']=model_backbone_type
        state_dict = torch.load(args.checkpoint, map_location='cpu')["state_dict"]
        new_state_dict1 = {}
        new_state_dict2 = {}
        new_state_dict3 = {}
            
        for key, value in state_dict.items():
                new_key = re.sub(r'^module\.', '', key, count=1)
                new_state_dict1[new_key] = value
        if "resnet" in args.config:
                mapped_state_dict = map_keys(new_state_dict1,args.config)
        else:
                mapped_state_dict = new_state_dict1

        for key, value in mapped_state_dict.items():  
                new_key1 = key.replace("backbone.", "")                             
                new_state_dict2[new_key1] = value
        for key, value in new_state_dict2.items():  
                new_key2 = key.replace("unit.", "")                             
                new_state_dict3[new_key2] = value
        model.load_state_dict(new_state_dict3,strict=True) 
        if eval(name)["backbone"]['type'] != 'ResNetLifItout_MP':
            network = cfg['model']["backbone"]['type'] + '/model_{}'.format(i)
            print(network)
        else:
            opath = './model_files/classification/'
            config_name_list = filename.split('/')[-1].split('-')
            dataset_key = config_name_list[3].split('.')[0].strip('_mp')
            network = config_name_list[0].capitalize() + datasets_abbr[dataset_key] + config_name_list[1].capitalize() + '_MP' + '/model_{}'.format(i)
        _base_ = os.path.join(opath, network)
        print('model_{} generated file path: {}'.format(i, _base_))
        in_size = [[]]
        in_size[0].append(models_compile_inputshape[i])
        model_operate.run(model.eval(), in_size, out_path=f'{_base_}')

if __name__ == '__main__':
    main()