###
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SeqClif5Fc2DmItout',
        timestep=20, input_channels=2, h=128, w=128, nclass=10, cmode='analog', amode='mean', noise=0, soma_params='all_share',
        neuron='lif', # neuron mode: 'lif' or 'lifplus'
        neuron_config=None # neron configs: 1.'lif': neuron_config=None; 2.'lifplus': neuron_config=[input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay], eg.[1,False,0,0,0,0]
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

###

dataset_type = 'DvsMnist'
train_pipeline = [
    dict(type='RandomCropVideo', size=128, padding=4),
    dict(type='ToTensorType', keys=['img'], dtype='float32'),  # ImageToTensor
]
test_pipeline = [
    dict(type='ToTensorType', keys=['img'], dtype='float32'),  # ImageToTensor
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,
    data_root='./data',
    train=dict(
        pipeline=train_pipeline
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    ))

###

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
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
