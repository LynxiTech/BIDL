###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SeqClifplus3Flifplus2DgItout',
        timestep=60, input_channels=2, h=40, w=40, nclass=11, cmode='spike', fmode='spike', amode='mean', noise=1e-3,
        soma_params='all_share',
        neuron='lifplus', # neuron mode: 'lif' or 'lifplus'
        neuron_config=[1,False,0,0,0,0] # neron configs: 1.'lif': neuron_config=None; 2.'lifplus': neuron_config=[input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay], eg.[1,False,0,0,0,0]
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

###

dataset_type = 'RgbGesture'
train_pipeline = [
    dict(type='RandomCropVideoDict', size=40, padding=4),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),  # ImageToTensor
    dict(type='ToTensor', keys=['gt_label']),
]
test_pipeline = [
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),  # ImageToTensor
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='./data/rgbgesture/',
        ann_file='train.pkl',
        pipeline=train_pipeline,
        shape=(40, 40)
    ),
    val=dict(
        type=dataset_type,
        data_prefix='./data/rgbgesture/',
        ann_file='val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        shape=(40, 40)
    ),
    test=dict(
        type=dataset_type,
        data_prefix='./data/rgbgesture/',
        ann_file='val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        shape=(40, 40)
    ))


###

# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='MultiStepLR', milestones=[20, 40, 60, 80])
runner = dict(type='EpochBasedRunner', max_epochs=50)

###

checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

fp16 = dict(loss_scale=512.0)  # half-percision
    