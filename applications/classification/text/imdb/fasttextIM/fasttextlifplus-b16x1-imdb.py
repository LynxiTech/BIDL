###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FastTextlifplusItout',
        timestep=500,
        vocab_size=1001, embedding_dim=16, hidden_dim=256, cmode='analog', amode='mean', noise=1e-3,soma_params='all_share',
        neuron='lifplus', # neuron mode: 'lif' or 'lifplus'
        neuron_config=[1,False,0,0,0,0] # neron configs: 1.'lif': neuron_config=None; 2.'lifplus': neuron_config=[input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay], eg.[1,False,0,0,0,0]
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss=dict(type='CrossEntropyLoss', use_sigmoid=True,loss_weight=1.0),
        topk=(1,),
        cal_acc=True
    ))

###

dataset_type = 'imdb'
train_pipeline = [    
    dict(type='ToTensorTypeDict', keys=['img'], dtype='int32'),  # ImageToTensor
    dict(type='ToTensor', keys=['gt_label']),
]
val_pipeline = [
    dict(type='ToTensorTypeDict', keys=['img'], dtype='int32'),  # ImageToTensor
]
test_pipeline = [
    dict(type='ToTensorTypeDict', keys=['img'], dtype='int64'),  # ImageToTensor
    dict(type='ToOneHot', keys=['img'],param=1001),  # XXX ImageToTensor
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='./data/imdb/train.npy',
        ann_file='./data/imdb/train_label.npy',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='./data/imdb/test.npy',
        ann_file='./data/imdb/test_label.npy',
        pipeline=val_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_prefix='./data/imdb/test.npy',
        ann_file='./data/imdb/test_label.npy',
        pipeline=test_pipeline,
        test_mode=True
    ))

###


optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='MultiStepLR', milestones=[10, 25, 35])
runner = dict(type='EpochBasedRunner', max_epochs=50)


###

checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

fp16 = dict(loss_scale=512.0)  # half-percision
