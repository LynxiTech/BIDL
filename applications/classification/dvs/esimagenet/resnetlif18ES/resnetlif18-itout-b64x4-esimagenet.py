###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetLifItout',
        timestep = 8,
        depth=18, nclass=1000,
        input_channels = 2,
        h=224,w=224,
        down_t=[1, 'avg'],
        cmode = 'analog',
        noise=0.001,
        soma_params='all_share'
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

img_norm_cfg = dict(
    mean=[118.656525, 111.0543, 106.928955], std=[60.48212, 61.974182, 60.785057],
    to_rgb=False
)
dataset_type = 'ESImagenet'
train_pipeline = [
    dict(type='LoadNumpy'),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),
    dict(type='ToTensor', keys=['gt_label']),
]
test_pipeline = [
    dict(type='LoadNumpy'),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),  
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='./data/ESImagenet/train',
        ann_file='./data/ESImagenet/trainlabel.csv',
        pipeline=train_pipeline,
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        data_prefix='./data/ESImagenet/val',
        ann_file='./data/ESImagenet/vallabel.csv',
        pipeline=test_pipeline,
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        data_prefix='./data/ESImagenet/val',
        ann_file='./data/ESImagenet/vallabel.csv',
        pipeline=test_pipeline,
        test_mode=False
    ),
)

###

optimizer = dict(type='SGD', lr=0.03, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='MultiStepLR', milestones=[30, 60, 90])
runner = dict(type='EpochBasedRunner', max_epochs=25)

###

checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# half-percision
fp16 = dict(loss_scale=512.0)

# find_unused_parameters = True
