'''
© 2022 Lynxi Technologies Co., Ltd. All rights reserved.
* NOTICE: All information contained here is, and remains the property of Lynxi. This file can not be copied or distributed without the permission of Lynxi Technologies Co., Ltd.
'''

import os
import pickle as pk

import elephant as el
import matplotlib.pyplot as plt
import neo
import numpy as np
import quantities as qn
import scipy.stats as sps
import sklearn as skl
import time
plt.switch_backend('agg')


def check_hist_data(xs: np.ndarray, hist: np.ndarray):
    assert all([isinstance(_, np.ndarray) for _ in (xs, hist)])
    assert all([len(_.shape) == 1 for _ in (xs, hist)])
    flag = not any([np.nan in xs, np.nan in hist, xs.shape[0] == 0, hist.shape[0] == 0])
    return flag


def calc_histogram(data: np.ndarray, xmin: float, xmax: float, n_smpl: int, smooth: bool):
    """Generate smoothed histogram from ``data``, with fallback measures."""
    assert len(data.shape) == 1 and data.shape[0] > 0

    def origin_histogram_np():
        """Sometimes smoothing operation fails, then use this impl."""
        hist, bins = np.histogram(data, n_smpl, (xmin, xmax))
        xs = (bins[:-1] + bins[1:]) / 2
        return xs, hist

    def smooth_histogram_sp():
        """This impl is very concise but its curve seems not that smooth."""
        kde = sps.kde.gaussian_kde(data, 'silverman')  # scott
        xs, d = np.linspace(xmin, xmax, n_smpl, retstep=True)
        hist = kde(xs)
        return xs, hist

    def smooth_histogram_skl():
        """The following impl may encounter some excetpions."""
        assert len(data.shape) == 1
        # Silverman rule for kernel density bandwidth
        std, iqr = np.std(data), sps.iqr(data)
        bw = 0.9 * min(std, iqr / 1.34) * len(data) ** (-0.2)
        # instantiate and fit the KDE model
        kde = skl.neighbors.KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(data[:, None])
        # score_samples returns the log of the probability density
        xs, d = np.linspace(xmin, xmax, n_smpl, retstep=True)
        prob = np.exp(kde.score_samples(xs[:, None]))
        # normalize
        hist = prob / np.sum(prob) / d
        return xs, hist

    if smooth is True:
        try:
            print('Calculate histogram using SciKit-Learn.')
            return smooth_histogram_skl()
        except:
            try:
                print('Fallback to SciPy.')
                return smooth_histogram_sp()
            except:
                raise NotImplementedError
    else:
        try:
            print('Calculate histogram using NumPy.')
            return origin_histogram_np()
        except:
            raise NotImplementedError


def draw_raster(
        spike_pair, save_path, save_raw=False,
        sim_time=1, resolut=.0001, n_smpl_max=(1080, 1920)
):
    moments, neurons = spike_pair
    num_max = sim_time / resolut
    slice_t = int(max([num_max / _ for _ in n_smpl_max]))
    print(f'slice_t={slice_t}')
    if slice_t > 1:
        moments = moments[::slice_t]
        neurons = neurons[::slice_t]
    __save_raw_or_fig(
        save_raw,
        title='Raster',
        xym=[moments, neurons, ',' if slice_t > 1 else '.'], label='spike',
        xlabel='time (ms)', ylabel='neuron id',
        text='',
        figfile=os.path.join(save_path, 'raster')
    )

    return moments, neurons


def draw_fireprob(
        spike_dict, save_path, duration, save_raw=False,
        n_smpl=1024, smooth=True
):
    n_spikes = [len(_) for _ in spike_dict.values()]
    spike_rates = np.array(n_spikes) / duration
    try:
        xs, hist = calc_histogram(spike_rates, 0, spike_rates.max(), n_smpl, smooth)
    except:
        xs, hist = np.array([]), np.array([])
    __save_raw_or_fig(
        save_raw,
        title='Fire Probability Density Distribution',
        xym=[xs, hist], label='fire prob.',
        xlabel='rate (spike/ms)', ylabel='',
        text='' if check_hist_data(xs, hist) else 'NO VALID DATA',
        figfile=os.path.join(save_path, 'fireprob')
    )

    return xs, hist


def draw_cvisi(
        spike_dict, save_path, save_raw=False,
        xmin=0.0, xmax=1.5, n_smpl=1024, smooth=True
):
    cvs = []
    for train in spike_dict.values():
        if len(train) > 1:
            isi = el.statistics.isi(train)  # inter-spike intervals
            cv = el.statistics.cv(isi)  # coefficient of variation
            if not np.isnan(cv):
                cvs.append(cv)
    try:
        xs, hist = calc_histogram(np.array(cvs), xmin, xmax, n_smpl, smooth)
    except:
        xs, hist = np.array([]), np.array([])
    __save_raw_or_fig(
        save_raw,
        title='CV of ISI',
        xym=[xs, hist], label='cv-isi',
        xlabel='cv of isi', ylabel='',
        text='' if check_hist_data(xs, hist) else 'NO VALID DATA',
        figfile=os.path.join(save_path, 'cvisi')
    )

    return xs, hist


def draw_corrcoef(
        spike_dict, save_path, t_start, t_stop, save_raw=False,
        xmin=-1, xmax=1, n_smpl=1024, bin_size=0.002, n_neur_max=4096, smooth=True
):
    assert (t_stop - t_start) / bin_size >= 2, f'Selected simulation time must be greater than {bin_size * 2 * 1000}ms!'
    assert (spike_dict ), f"there must have spike when draw corrcoef "
    trains = []
    keys = list(spike_dict.keys())
    keys.sort()
    _trains_ = []
    for neuron in keys[:n_neur_max]:
        train = spike_dict[neuron]
        spike_train = neo.core.SpikeTrain(train * qn.s, t_stop=t_stop * qn.s)
        trains.append(spike_train)
        _trains_.append(train)
    
    trains_bin = el.conversion.BinnedSpikeTrain(
        trains, bin_size * qn.s, None, t_start * qn.s, t_stop * qn.s
    )
 
    cc_mat = el.spike_train_correlation.correlation_coefficient(trains_bin)
    ccs_ = cc_mat[np.where(~np.eye(len(trains), dtype='bool') & (cc_mat >= xmin) & (cc_mat <= xmax))]
    try:
        xs, hist = calc_histogram(ccs_.flatten(), xmin, xmax, n_smpl, smooth)
    except:
        xs, hist = np.array([]), np.array([])
    __save_raw_or_fig(
        save_raw,
        title='Pearson Correlation',
        xym=[xs, hist], label='Pearson coorelation',
        xlabel='corr. coef.', ylabel='',
        text='' if check_hist_data(xs, hist) else 'NO VALID DATA',
        figfile=os.path.join(save_path, 'corrcoef')
    )
    
    return xs, hist


def __save_raw_or_fig(
        save_raw: bool,
        title: str, xym: list, label: str, xlabel: str, ylabel: str, text: str, figfile: str,
        figsize=(16, 9), fontsize=18, dpi=100
):
    viz_dict = dict(
        figure=([1], dict(figsize=figsize)),
        cla=([], dict()),
        title=([title], dict(fontsize=int(fontsize * 1.25))),
        plot=(xym, dict(label=label)),
        xlabel=([xlabel], dict(fontsize=fontsize)),
        ylabel=([ylabel], dict(fontsize=fontsize)),
        text=([0, 0, text], dict(fontsize=fontsize * 2, ha='center', va='center')),
        savefig=([figfile], dict(dpi=dpi))
    )
    if save_raw is True:
        with open(f'{figfile}.pkl', 'wb') as f:
            pk.dump(viz_dict, f)
    else:
        [eval(f'plt.{k}')(*v[0], **v[1]) for k, v in viz_dict.items()]
    print("save {} figure to path:".format(title),os.path.join(figfile.split('/')[0], \
                "{}.png".format(figfile.split('/')[-1])))


def draw_multiple(vdicts: list, labels: list, save_path: str):
    assert len(vdicts) >= 2
    assert all([isinstance(_, dict) for _ in vdicts])
    markers = '.12348spP*hH+xXDd|_,ov^<>'  # '.,ov^<>12348spP*hH+xXDd|_'
    assert len(vdicts) == len(labels) <= len(markers)

    exec_final = dict()
    for i, (viz_dict, label, marker) in enumerate(zip(vdicts, labels, markers)):
        if i == 0:
            exec_final['legend'] = ([], dict(loc='upper right'))
            exec_final['savefig'] = ([save_path], viz_dict.pop('savefig')[1])
        else:
            [viz_dict.pop(_) for _ in ('figure', 'cla', 'title', 'xlabel', 'savefig')]

        kwargs = dict(label=label, marker=marker)
        
        if len(viz_dict['plot'][0]) > 2:
            viz_dict['plot'][0][-1] = ""
            #kwargs.pop('marker')        
        viz_dict['plot'] = (viz_dict['plot'][0], kwargs)        
        [eval(f'plt.{k}')(*v[0], **v[1]) for k, v in viz_dict.items()]

    [eval(f'plt.{k}')(*v[0], **v[1]) for k, v in exec_final.items()]

def draw_single(vdicts: list, labels: list, save_path: str):
    assert len(vdicts) == 1
    assert all([isinstance(_, dict) for _ in vdicts])
    markers = '.12348spP*hH+xXDd|_,ov^<>'  # '.,ov^<>12348spP*hH+xXDd|_'
    assert len(vdicts) == len(labels) <= len(markers)

    exec_final = dict()
    for i, (viz_dict, label, marker) in enumerate(zip(vdicts, labels, markers)):
        if i == 0:
            exec_final['legend'] = ([], dict(loc='upper right'))
            exec_final['savefig'] = ([save_path], viz_dict.pop('savefig')[1])
        else:
            [viz_dict.pop(_) for _ in ('figure', 'cla', 'title', 'xlabel', 'savefig')]

        kwargs = dict(label=label, marker=marker)
        if len(viz_dict['plot'][0]) > 2:
            viz_dict['plot'][0][-1] = ""
        #     kwargs.pop('marker')
        viz_dict['plot'] = (viz_dict['plot'][0], kwargs)

        [eval(f'plt.{k}')(*v[0], **v[1]) for k, v in viz_dict.items()]

    [eval(f'plt.{k}')(*v[0], **v[1]) for k, v in exec_final.items()]



def draw_plots(spike_pair, spike_dict, stime, resolut, save_path, save_raw, smooth):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    t1 = time.time()
    draw_raster(
        spike_pair, save_path, save_raw=save_raw,
        sim_time=stime, resolut=resolut, n_smpl_max=(1080, 1920),
    )
    print('dt1:', time.time() - t1)

    t2 = time.time()
    draw_fireprob(
        spike_dict, save_path, stime, save_raw=save_raw,
        n_smpl=256, smooth=smooth
    )
    print('dt2:', time.time() - t2)

    t3 = time.time()
    draw_cvisi(
        spike_dict, save_path, save_raw=save_raw,
        xmin=0, xmax=1.5, n_smpl=256, smooth=smooth
    )
    print('dt3:', time.time() - t3)

    t4 = time.time()
    draw_corrcoef(
        spike_dict, save_path, 0, stime, save_raw=save_raw,
        xmin=-1, xmax=1, n_smpl=256, bin_size=0.002, n_neur_max=256, smooth=smooth  # xmin=-0.05, xmax=0.15
    )
    print('dt4:', time.time() - t4)

    print('dt1~4:', time.time() - t1)  # i5-11230H: 0.67; i7-7700K: 0.89; E5-2640v4: 1.25
