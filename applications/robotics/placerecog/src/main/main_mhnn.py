
import torch.utils.model_zoo
import time
from src.model.mhnn import *
from src.tools.utils import *
from src.config.config_utils import *

import argparse
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

setup_seed(0)

parser = argparse.ArgumentParser(description='mhnn', argument_default=argparse.SUPPRESS)
parser.add_argument('--config_file', type=str, required=True, help='Configure file path')

def Pre_process(inputs):
    out0_zeros = torch.zeros_like(inputs[0])
    inputs[0] = torch.cat([inputs[0], out0_zeros, out0_zeros], 1)
    out1_zeros = torch.zeros_like(inputs[1])
    inputs[1] = torch.cat([inputs[1], out1_zeros, out1_zeros], 1)
    out3_zeros = torch.zeros_like(inputs[3])
    inputs[3] = torch.cat([inputs[3], out3_zeros, out3_zeros], 1)
    # print(inputs[0].shape)
    # print(inputs[1].shape)
    # print(inputs[2].shape)
    # print(inputs[3].shape)
    return inputs

def main_mhnn(options: argparse.Namespace):
    config = get_config(options.config_file, logger=log)['main']
    config = update_config(config, options.__dict__)

    cnn_arch = str(config['cnn_arch'])
    num_epoch = int(config['num_epoch'])
    batch_size = int(config['batch_size'])
    num_class = int(config['num_class'])
    rnn_num = int(config['rnn_num'])
    cann_num = int(config['cann_num'])
    reservoir_num = int(config['reservoir_num'])
    num_iter = int(config['num_iter'])
    spiking_threshold = float(config['spiking_threshold'])
    sparse_lambdas = int(config['sparse_lambdas'])
    lr = float(config['lr'])
    r = float(config['r'])

    ann_pre_load = get_bool_from_config(config, 'ann_pre_load')
    snn_pre_load = get_bool_from_config(config, 'snn_pre_load')
    re_trained = get_bool_from_config(config, 're_trained')

    seq_len_aps = int(config['seq_len_aps'])
    seq_len_gps = int(config['seq_len_gps'])
    seq_len_dvs = int(config['seq_len_dvs'])
    seq_len_head = int(config['seq_len_head'])
    seq_len_time = int(config['seq_len_time'])

    dvs_expand = int(config['dvs_expand'])
    expand_len = least_common_multiple([seq_len_aps, seq_len_dvs * dvs_expand, seq_len_gps])

    test_exp_idx = []
    for idx in config['test_exp_idx']:
        if idx != ',':
            test_exp_idx.append(int(idx))

    train_exp_idx = []
    for idxt in config['train_exp_idx']:
        if idxt != ',':
            train_exp_idx.append(int(idxt))

    data_path = str(config['data_path'])
    snn_path = str(config['snn_path'])
    hnn_path = str(config['hnn_path'])
    model_saving_file_name = str(config['model_saving_file_name'])

    w_fps = int(config['w_fps'])
    w_gps = int(config['w_gps'])
    w_dvs = int(config['w_dvs'])
    w_head = int(config['w_head'])
    w_time = int(config['w_time'])

    device_id = str(config['device_id'])

    normalize = torchvision.transforms.Normalize(mean=[0.3537, 0.3537, 0.3537],
                                                 std=[0.3466, 0.3466, 0.3466])

    train_loader = Data(data_path, batch_size=batch_size, exp_idx=train_exp_idx, is_shuffle=True,
                        normalize=normalize, nclass=num_class,
                        seq_len_aps=seq_len_aps, seq_len_dvs=seq_len_dvs, seq_len_gps=seq_len_gps,
                        seq_len_head=seq_len_head, seq_len_time = seq_len_time)

    test_loader = Data(data_path, batch_size=batch_size, exp_idx=test_exp_idx, is_shuffle=True,
                       normalize=normalize, nclass=num_class,
                       seq_len_aps=seq_len_aps, seq_len_dvs=seq_len_dvs, seq_len_gps=seq_len_gps,
                       seq_len_head=seq_len_head, seq_len_time = seq_len_time)

    mhnn = MHNN(device = device,
        cnn_arch = cnn_arch,
        num_epoch = num_epoch,
        batch_size = batch_size,
        num_class = num_class,
        rnn_num = rnn_num,
        cann_num = cann_num,
        reservoir_num = reservoir_num,
        spiking_threshold = spiking_threshold,
        sparse_lambdas = sparse_lambdas,
        r = r,
        lr = lr,
        w_fps = w_fps,
        w_gps = w_gps,
        w_dvs = w_dvs,
        w_head = w_head,
        w_time = w_time,
        seq_len_aps = seq_len_aps,
        seq_len_gps = seq_len_gps,
        seq_len_dvs = seq_len_dvs,
        seq_len_head = seq_len_head,
        seq_len_time = seq_len_time,
        dvs_expand = dvs_expand,
        expand_len = expand_len,
        train_exp_idx = train_exp_idx,
        test_exp_idx = test_exp_idx,
        data_path = data_path,
        snn_path = snn_path,
        hnn_path = hnn_path,
        num_iter = num_iter,
        ann_pre_load = ann_pre_load,
        snn_pre_load = snn_pre_load,
        re_trained = re_trained)

    mhnn.cann_init(np.concatenate((train_loader.dataset.data_pos[0],train_loader.dataset.data_head[0][:,1].reshape(-1,1)),axis=1))

    mhnn.to(device)

    optimizer = torch.optim.Adam(mhnn.parameters(), lr)

    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    record = {}
    record['loss'], record['top1'], record['top5'], record['top10'] = [], [], [], []
    best_test_acc1, best_test_acc5, best_recall, best_test_acc10 = 0., 0., 0, 0

    train_iters = iter(train_loader)
    iters = 0
    start_time = time.time()
    print(device)

    import torchmetrics

    best_recall = 0.
    # test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
    #
    # test_recall = torchmetrics.Recall(task="multiclass", average='none', num_classes=num_class)
    # test_precision = torchmetrics.Precision(task="multiclass", average='none', num_classes=num_class)

    test_acc = torchmetrics.Accuracy()

    test_recall = torchmetrics.Recall(average='none', num_classes=num_class)
    test_precision = torchmetrics.Precision(average='none', num_classes=num_class)

    for epoch in range(num_epoch):

        ###############
        ## for training
        ###############

        running_loss = 0.
        counts = 1.
        acc1_record, acc5_record, acc10_record = 0., 0., 0.
        while iters < num_iter:
            mhnn.train()
            optimizer.zero_grad()

            try:
                inputs, target = next(train_iters)
            except StopIteration:
                train_iters = iter(train_loader)
                inputs, target = next(train_iters)

            inputs_ = Pre_process(inputs)

            # print(inputs_[0].shape)
            # print(inputs_[1].shape)
            # print(inputs_[2].shape)
            # print(inputs_[3].shape)

            outputs, outs = mhnn(inputs_, epoch=epoch)

            class_loss = criterion(outputs, target.to(device))

            sparse_loss = 0.
            w_module = (w_fps, w_dvs, w_gps, w_gps)
            for (i, l) in enumerate(outs):
                sparse_loss += (l.mean() - r)**2 * w_module[i]

            loss = class_loss + sparse_lambdas * sparse_loss

            loss.backward()

            optimizer.step()

            running_loss += loss.cpu().item()

            acc1, acc5, acc10 = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
            acc1, acc5, acc10 = acc1 / len(outputs), acc5 / len(outputs), acc10 / len(outputs)

            acc1_record += acc1
            acc5_record += acc5
            acc10_record += acc10

            counts += 1
            iters += 1.
        iters = 0

        record['loss'].append(loss.item())
        print('\n\nTime elaspe:', time.time() - start_time, 's')
        print(
            'Training epoch %.1d, training loss :%.4f, sparse loss :%.4f, training Top1 acc: %.4f, training Top5 acc: %.4f' %
            (epoch, running_loss / (num_iter), sparse_lambdas * sparse_loss, acc1_record / counts, acc5_record / counts))

        lr_schedule.step()
        start_time = time.time()

        ##############
        ## for testing
        ##############

        running_loss = 0.
        mhnn.eval()

        with torch.no_grad():

            acc1_record, acc5_record, acc10_record = 0., 0., 0.
            counts = 1.

            for batch_idx, (inputs, target) in enumerate(test_loader):

                inputs_ = Pre_process(inputs)
                outputs, _ = mhnn(inputs_, epoch=epoch)

                loss = criterion(outputs.cpu(), target)

                running_loss += loss.item()
                acc1, acc5, acc10 = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
                acc1, acc5, acc10 = acc1 / len(outputs), acc5 / len(outputs), acc10 / len(outputs)
                acc1_record += acc1
                acc5_record += acc5
                acc10_record += acc10

                counts += 1
                outputs = outputs.cpu()

                test_acc(outputs.argmax(1), target)
                test_recall(outputs.argmax(1), target)
                test_precision(outputs.argmax(1), target)

                compute_num = 100
                if batch_idx % compute_num == 0 and batch_idx > 0:
                    #print('time:', (time.time() - start_time) / compute_num)
                    start_time = time.time()

        total_acc = test_acc.compute().mean()
        total_recall = test_recall.compute().mean()
        total_precison = test_precision.compute().mean()

        test_precision.reset()
        test_recall.reset()
        test_acc.reset()

        acc1_record = acc1_record / counts
        acc5_record = acc5_record / counts
        acc10_record = acc10_record / counts

        record['top1'].append(acc1_record)
        record['top5'].append(acc5_record)
        record['top10'].append(acc10_record)


        print('Current best Top1: ', best_test_acc1, 'Best Top5:', best_test_acc5)

        if epoch > 4:
            if best_test_acc1 < acc1_record:
                best_test_acc1 = acc1_record
                print('Achiving the best Top1, saving...', best_test_acc1)

            if best_test_acc5 < acc5_record:
                # best_test_acc1 = acc1_record
                best_test_acc5 = acc5_record
                print('Achiving the best Top5, saving...', best_test_acc5)

            if best_recall < total_recall:
                # best_test_acc1 = acc1_record
                best_recall = total_recall
                print('Achiving the best recall, saving...', best_recall)

            if best_test_acc10 < acc10_record:
                best_test_acc10 = acc10_record

            state = {
                'net': mhnn.state_dict(),
                'snn': mhnn.snn.state_dict(),
                'record': record,
                'best_recall': best_recall,
                'best_acc1': best_test_acc1,
                'best_acc5': best_test_acc5,
                'best_acc10': best_test_acc10
            }
            if not os.path.isdir('../../checkpoint'):
                os.mkdir('../../checkpoint')
            torch.save(state, '../../checkpoint/' + model_saving_file_name + '.t7')


if __name__ == '__main__':
    options, unknowns = parser.parse_known_args()
    main_mhnn(options)








