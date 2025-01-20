###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SeqClifplus3Fc3LcItout',
        timestep=8, input_channels=1, h=32, w=32, nclass=2, cmode='analog', amode='mean', noise=0, soma_params='all_share',
        neuron='lifplus', # neuron mode: 'lif' or 'lifplus'
        neuron_config=[1,False,0,0,0,0] # neron configs: 1.'lif': neuron_config=None; 2.'lifplus': neuron_config=[input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay], eg.[1,False,0,0,0,0]
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk= (1, ),
        cal_acc=True
    ))

###

dataset_type = 'Luna16Cls32'
train_pipeline = [
    # dict(type='RandomCropVideo', size=32, padding=4),  # XXX
    # dict(type='Rotate', angle=3.0),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),  # XXX ImageToTensor
    dict(type='ToTensor', keys=['gt_label']),
]
test_pipeline = [
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),  # XXX ImageToTensor
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='./data/luna16cls/train3/',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='./data/luna16cls/test3/',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_prefix='./data/luna16cls/test3/',
        pipeline=test_pipeline,
        test_mode=True
    ))

###

optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001, nesterov=True)
# optimizer = dict(type='Adam', lr=0.03, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='MultiStepLR', milestones=[10, 16])
warmup=dict(warmup_ratio=1e-2, warmup_iters=500)
runner = dict(type='EpochBasedRunner', max_epochs=20)

###

checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

fp16 = dict(loss_scale=512.0)  # half-percision
