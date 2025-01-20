'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import torch
import numpy as np
from torch import ops, nn
import time
from neurontypes import lif_neuron, izhikevich, Adex, MultiCompartment, HodHux
import pickle as pk
import random
import json
from multi_cluster_modules import Population, Projection, SpikeInput, PoissonInput,STDPLearner
import matplotlib.pyplot as plt
plt.switch_backend('agg')
DATA_DIR = os.path.join(os.getcwd(), 'multi_cluster_config/')
CONFIG_PATH = os.path.join(DATA_DIR, 'stdp.json')

sim_dict = {
    # Resolution of the simulation (in ms).
    'sim_resolution': 0.1,
    # Delay
    'delay': 0.1
}


class Verify(object):
    def __init__(self, config_path):
        self.config_path = config_path
        with open(self.config_path, 'r') as f:
            config_json = json.load(f)
        self.populations = config_json["population"]
        self.projections = config_json["projection"]
        if "spike_input" in config_json.keys():
            self.spike_inputs = config_json["spike_input"]

def parse_args():
    parser = argparse.ArgumentParser(description='neuron model simulation')   
    parser.add_argument('--use_lyngor', type=int, help='use lyngor flag: 1 or 0, means if use lyngor or not')
    parser.add_argument('--use_legacy', type=int, help='use legacy flag: 1 or 0, means if use legacy or not')
    parser.add_argument('--use_gpu', type=int, help='use gpu flag: 1 or 0, means if use gpu or not')
    parser.add_argument('--plot', type=int, help='draw plot flag: 1 or 0, means if draw plot or not')
    parser.add_argument('--device', default=0, type=int,help='apu device id')
    args = parser.parse_args()
    return args



class MultiClusterNetwork(torch.nn.Module):
    def __init__(self, sim_dict=sim_dict, on_apu=True):
        super(MultiClusterNetwork, self).__init__()
        self.on_apu = on_apu
        self.t = 0.0
        self.proj_iaf = None
        self.pops = None
        self.spike_inputs = None
        self.poisson_inputs = None
        self.neuron_index = 0
        self.sim_dict = sim_dict
        self.config = Verify(CONFIG_PATH)
        self.setup(on_apu=on_apu)

    def setup(self, on_apu=False):
        self.create_populations(on_apu=on_apu)
        self.create_connections(on_apu=on_apu)
        if hasattr(self.config, "spike_inputs"):
            self.create_spikeinputs()
        self.create_poissoninputs()

    def create_populations(self, on_apu=False):
        pops = []
        for i, pop in enumerate(self.config.populations):
            # create population
            neuron_num = int(pop['neuron_index'][1] - pop['neuron_index'][0]) + 1
            population = Population(num=neuron_num,
                                    resolution=self.sim_dict['sim_resolution'],
                                    delay=self.sim_dict['delay'],
                                    neuron_id=self.neuron_index,
                                    pop_index=i,
                                    neuron_param=pop["params"],
                                    ex_inh_type=pop['ex_inh_type'],
                                    on_apu=on_apu)
            pops.append(population)
            self.neuron_index += neuron_num
        self.pops = torch.nn.ModuleList(pops)
        pops.clear()

    def create_connections(self, on_apu=False):
        proj_iaf = []
        for i, proj in enumerate(self.config.projections):
            # source population index, target population index
            source_id, target_id = list(map(int, proj['proj'].split('_')))
            source_pop, target_pop = self.pops[source_id], self.pops[target_id]
            if "learning" in proj and proj["learning"]=="True":
                learning = True
            else:
                learning = False
            proj_iaf.append(Projection(source_pop, target_pop, sparse_ratio=proj['sparse_ratio'], on_apu=on_apu,learning=learning))
        self.proj_iaf = torch.nn.ModuleList(proj_iaf)
        proj_iaf.clear()

    def create_spikeinputs(self):
        spike_inputs = []
        for i, spikeinput in enumerate(self.config.spike_inputs):
            target_pop = self.pops[int(spikeinput["pos_pop"])]
            spike_inputs.append(SpikeInput(len(spikeinput["spike_input"]),
                                           target_pop,
                                           sparse_ratio=spikeinput["sparse_ratio"]))

        self.spike_inputs = torch.nn.ModuleList(spike_inputs)
        spike_inputs.clear()

    def create_poissoninputs(self):
        poisson_inputs = []
        for i, pop in enumerate(self.config.populations):
            if "poisson_rate" in pop:
                target_pop = self.pops[i]
                poisson_inputs.append(PoissonInput(pop["neuron_number"], target_pop))

        if len(poisson_inputs) != 0:
            self.poisson_inputs = torch.nn.ModuleList(poisson_inputs)
            poisson_inputs.clear()

    def forward(self, dc_inpt):
        for proj in self.proj_iaf:           
            weight = proj.forward()           

        if hasattr(self.config, "spike_inputs"):
            for i, spike_input in enumerate(self.spike_inputs):
                inpt = (dc_inpt[len(self.pops) + i][0, :spike_input.input_num]).unsqueeze(dim=0)
                spike_input.forward(inpt)

        if self.spike_inputs is not None:
            poisson_index = len(self.pops) + len(self.spike_inputs)
        else:
            poisson_index = len(self.pops)
        if self.poisson_inputs is not None:
            for i, poisson_input in enumerate(self.poisson_inputs):
                inpt = (dc_inpt[poisson_index + i][0, :poisson_input.pos_popu.num]).unsqueeze(dim=0)
                poisson_input.forward(inpt)

        # each population update
        results = []
        for i, pop in enumerate(self.pops):            
            inpt = (dc_inpt[i][:,:pop.num]).unsqueeze(dim=0)           
            res = pop.forward(currents=inpt)
            if self.max_pop_neuron_num > pop.num:
                padding_shape = (1, self.max_pop_neuron_num - pop.num)
                res = torch.cat((res, torch.zeros(padding_shape,dtype=torch.float)), dim=1)
            results.append(res)
        return_res = torch.cat(results)
        
        return return_res,weight



def main():
    args = parse_args()    
    assert args.use_lyngor in (0, 1), 'use_lyngor must in (0, 1)'
    assert args.use_legacy in (0, 1), 'use_legacy must in (0, 1)'
    assert args.use_gpu in (0, 1), 'use_gpu must in (0, 1)'
    assert args.plot in (0, 1), 'plot must in (0, 1)'
    assert isinstance(args.device, int), 'apu chip id must be int'
    chip_id = args.device
    USE_LYNGOR = True if args.use_lyngor == 1 else False
    USE_LEGACY = True if args.use_legacy == 1 else False
    USE_GPU = True if args.use_gpu == 1 else False
    PLOT = True if args.plot == 1 else False


 
    b, n, t, c = 1, 1, 104, 2

    configs = Verify(CONFIG_PATH)
    pop_num = len(configs.populations)
    if hasattr(configs, "spike_inputs"):
        spike_input_num = len(configs.spike_inputs)
    neuron_nums = 0
    max_pop_neuron_num = 0
    for i in range(pop_num):
        pop_neuron_num = configs.populations[i]['neuron_number']
        if pop_neuron_num > max_pop_neuron_num:
            max_pop_neuron_num = pop_neuron_num
        neuron_nums += pop_neuron_num
    print("neurons: ", neuron_nums)
    if hasattr(configs, "spike_inputs"):
        for i in range(spike_input_num):
            inpt_spike_num = len(configs.spike_inputs[i]["spike_input"])
            if inpt_spike_num > max_pop_neuron_num:
                max_pop_neuron_num = inpt_spike_num

    poisson_input_num = 0
    for k in range(pop_num):
        if "poisson_rate" in configs.populations[k]:
            poisson_input_num += 1

    p_eye = torch.eye(max_pop_neuron_num)*100.
    p_in = torch.zeros(max_pop_neuron_num,max_pop_neuron_num) 
    p = torch.cat((p_eye,p_in),dim=-1)
    p_pre = torch.zeros(max_pop_neuron_num,max_pop_neuron_num*2, dtype=torch.float32) 
    p_pre[max_pop_neuron_num//2,:] = 100.                    
    inputs = torch.cat((p_pre,p),dim=1)
    if not os.path.isdir("./res"):
        os.makedirs("./res")
    
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]   

    if USE_LYNGOR:
        random.seed(456)
        torch.manual_seed(1234)
        rng = np.random.default_rng(seed=1234)
        model = MultiClusterNetwork(on_apu=True)        
        model.max_pop_neuron_num = max_pop_neuron_num        
        #opath = os.path.join(os.getcwd(), 'model_config/stdp/')
        opath = os.path.join(os.path.join(ROOT, "../../../model_files/neuralsim"),  "stdp")  
        # import sys
        # sys.path.append("../../..")
        from lynadapter.lyn_sdk_model import ApuRun
        
        if hasattr(configs, "spike_inputs"):
            in_size = [((1 * (pop_num + spike_input_num + poisson_input_num), c, max_pop_neuron_num),)] 
        else:
            in_size = [((1 * (pop_num + poisson_input_num), c, max_pop_neuron_num),)]
        if not USE_LEGACY:
            from lynadapter.lyn_compile import model_compile
            input_type = "float32"
            model_compile(model,opath,in_size,input_type=input_type) 
            
        model_path = os.path.join(opath, "Net_0")

       
        arun = ApuRun(chip_id, model_path,t)
        apu_result = []
        weight_all =[]
        t0 = time.time()
        for bi in range(n):
            inputs_all = []
            for i in range(t):               
                # 1. dc input
                all_pops_inputs = tuple(torch.zeros(b, c, max_pop_neuron_num, dtype=torch.float32) for i in range(pop_num))
                data = torch.cat(all_pops_inputs)
                data = data.numpy().astype(np.float32)
                if i<max_pop_neuron_num:
                    datas = inputs[i, :].reshape(2,2,max_pop_neuron_num).numpy().astype(np.float32)
                    inputs_all.append(datas)
                    #arun.load_data(datas)
                else:                
                    #arun.load_data(data)
                    inputs_all.append(data)
            inputs_all = np.array(inputs_all) 

            arun.run(inputs_all)
        
        
        apu_result = np.array(arun.get_output()[0][0]).reshape(t, 1, pop_num, max_pop_neuron_num)
        weight_all = np.array(arun.get_output()[0][1]).reshape(t,max_pop_neuron_num)               
        
        # apu_result.append(output)
        # weight_all.append(weight)

        t1 = time.time()
        print("apu: timesteps/s: ", n * t / (t1 - t0))
           
        weight = np.array(np.stack(weight_all))
        mask = (weight!=0.)        
        weight = np.insert(weight[mask],0,0.0)   
        t_axis = np.arange(-1*max_pop_neuron_num//2,max_pop_neuron_num//2)      
        plt.plot(t_axis, weight)      
        plt.ylabel('delta_weight', rotation=90)
        plt.xlabel('PostSpikeTime-PreSpikeTime', rotation=0, labelpad=10)
        plt.grid(True)      
        plt.gcf().subplots_adjust(left=0.18)        
        plt.show()
        plt.savefig('res/apu_stdp.png')

    if USE_GPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        random.seed(456)
        torch.manual_seed(1234)
        rng = np.random.default_rng(seed=1234)
        model = MultiClusterNetwork(on_apu=False).to(device)        
        model.max_pop_neuron_num = max_pop_neuron_num
        gpu_result = []
        gpu_weight =[]
        t0 = time.time()
        for bi in range(n):
            for i in range(t):                
                all_pops_inputs = tuple(torch.zeros(b, c, max_pop_neuron_num, dtype=torch.float32) for i in range(pop_num))                   
                xi = torch.cat(all_pops_inputs)
                if i<max_pop_neuron_num:
                    s,weight = model(inputs[i, :].reshape(2,2,max_pop_neuron_num))
                else:
                    s,weight = model(xi)
                s = s.reshape(-1)
                gpu_weight.append(weight)
                if torch.cuda.is_available():
                    gpu_result.append(s.cpu().numpy()[None][None]
                                      if model.__class__.__name__ == 'MultiClusterNetwork' else s.cpu().numpy())
                else:
                    gpu_result.append(s.numpy()[None][None]
                                      if model.__class__.__name__ == 'MultiClusterNetwork' else s.numpy())
        t1 = time.time()
        print("gpu: timesteps/s: ", n * t / (t1 - t0))          
        weight = np.array(np.stack(gpu_weight))
        mask = (weight!=0.)        
        weight = np.insert(weight[mask],0,0.0)       
        t_axis = np.arange(-1*max_pop_neuron_num//2,max_pop_neuron_num//2)       
        plt.plot(t_axis, weight)        
        plt.ylabel('delta_weight', rotation=90)
        plt.xlabel('PostSpikeTime-PreSpikeTime', rotation=0, labelpad=10)
        plt.grid(True)
        plt.gcf().subplots_adjust(left=0.18)        
        plt.show()
        plt.savefig('res/gpu_stdp.png')
    

    if USE_GPU and USE_LYNGOR:
        fire_apu = 0
        fire_gpu = 0

        for bi in range(n * t):
            apu_b = apu_result[bi]
            gpu_b = gpu_result[bi]
            fire_apu += np.sum(apu_b == 1)
            fire_gpu += np.sum(gpu_b == 1)
        
        fire_total_num = n * t * pop_num * max_pop_neuron_num
        if fire_apu == 0 and fire_gpu == 0:
            fire_rate_acc = 0.0
        elif fire_gpu == 0:
            fire_rate_acc = (abs(fire_apu - fire_gpu) / fire_total_num) / (fire_apu / fire_total_num)
        else:
            fire_rate_acc = (abs(fire_apu - fire_gpu) / fire_total_num) / (fire_gpu / fire_total_num)
        print(f"fire rate difference: {round(fire_rate_acc * 100., 3)}%")

    if PLOT:
        from snnviz import draw, utils, load
        assert (USE_GPU or USE_LYNGOR), f"plot must have gpu or apu result"
        resolut = 0.1
        stime = t * resolut
        visual_fold = 'res/'
        spike_pairs = []
        spike_dicts = []
        output_folds = []
        spike_labels = []

        if USE_LYNGOR:
            spike_pair_apu = utils.spike_mask_to_pair(np.stack(apu_result, axis=1), resolut)
            spike_pairs.append(spike_pair_apu)
            spike_dicts.append(load.spike_pair_to_dict(spike_pair_apu))
            output_folds.append('res/spikes_apu/')
            spike_labels.append('spikes_apu')
        if USE_GPU:
            spike_pair_gpu = utils.spike_mask_to_pair(np.stack(gpu_result, axis=1), resolut)
            spike_pairs.append(spike_pair_gpu)
            spike_dicts.append(load.spike_pair_to_dict(spike_pair_gpu))
            output_folds.append('res/spikes_gpu/')
            spike_labels.append('spikes_gpu')

        for spair, sdict, ofold in zip(spike_pairs, spike_dicts, output_folds):
            draw.draw_plots(spair, sdict, stime, resolut, ofold, True, True)

        outputs = [[os.path.join(_, __) for __ in os.listdir(_) if __.endswith('.pkl')] for _ in output_folds]
        [_.sort() for _ in outputs]

        edict = {}
        if USE_LYNGOR and USE_GPU:
            for pfile_apu, pfile_gpu in zip(*outputs):
                title = os.path.split(pfile_apu)[-1][:-4]
                print(f'draw ``{title}``...')

                vdicts = []
                for pfile in (pfile_apu, pfile_gpu):
                    with open(pfile, 'rb') as f:
                        vdicts.append(pk.load(f))

                if title != 'raster':
                    ys_apu, ys_gpu = [_['plot'][0][1] for _ in vdicts]
                    try:
                        err = utils.calc_norm_diff(ys_gpu, ys_apu)  # Take ``gpu`` as ``gt``!
                    except:
                        import math
                        err = math.nan
                    edict[title] = err
                    print("{} norm diff ".format(title), err)

                ofile = os.path.join(visual_fold, f'{title}.png')
                draw.draw_multiple(vdicts, spike_labels, ofile)
        else:
            show_str = "lyngor" if USE_LYNGOR else "gpu"
            print(f"only plot {show_str}, no norm calculated")
            for pfile in outputs[0]:
                title = os.path.split(pfile)[-1][:-4]
                print(f'draw ``{title}``...')

                vdicts = []
                with open(pfile, 'rb') as f:
                    vdicts.append(pk.load(f))

                ofile = os.path.join(visual_fold, f'{title}.png')
                draw.draw_single(vdicts, spike_labels, ofile)


if __name__ == '__main__':
    main()
