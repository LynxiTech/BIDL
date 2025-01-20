###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetLifItout',
        depth=18, nclass=27,
        timestep=16,h=112,w=112,input_channels=3,
        down_t=[1, 'avg'],
        noise=0.001,
        cmode = 'analog',
        soma_params='all_share',
        norm = dict(
       mean=[118.656525, 111.0543, 106.928955], std=[60.48212, 61.974182, 60.785057])
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
dataset_type = 'Jester20bn'
train_pipeline = [
    dict(type='LoadFramesInFolder', n_frame=16, down_t=(1.0, 2.0, 0.1), dropout=0.05, random=True),
    dict(type='CropPadByRatioVideo', croppad=None, allowed_ratio=[1, 4 / 3], min_padd=.1, max_padd=.2,
        pad_mode=['constant'] * 4),
    dict(type='CutOutVideo', area=0.2, fill=None, prob=0.4),
    dict(type='RandomResizedCropVideo', center=(0, 0), size=112, scale=(0.4, 1.0), ratio=(3 / 4, 4 / 3)),
    #dict(type='NormalizeVideo', **img_norm_cfg),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),
    dict(type='ToTensor', keys=['gt_label']),
]
test_pipeline = [
    dict(type='LoadFramesInFolder', n_frame=16, down_t=(1.5,), random=False),
    dict(type='CropPadByRatioVideo', croppad=None, allowed_ratio=[1, 4 / 3], min_padd=.1, max_padd=.2,
        pad_mode='constant'),
    dict(type='ResizeVideo', size=(112, -1)),
    dict(type='CenterCropVideo', crop_size=112),
    #dict(type='NormalizeVideo', **img_norm_cfg),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),
]

test_pipeline_gpu = [
    dict(type='LoadFramesInFolder', n_frame=16, down_t=(1.5,), random=False),
    dict(type='CropPadByRatioVideo', croppad=None, allowed_ratio=[1, 4 / 3], min_padd=.1, max_padd=.2,
        pad_mode='constant'),
    dict(type='ResizeVideo', size=(112, -1)),
    dict(type='CenterCropVideo', crop_size=112),
    #dict(type='NormalizeVideo', **img_norm_cfg),
    dict(type='ToTensorTypeDict', keys=['img'], dtype='float32'),
]

data = dict(
    samples_per_gpu=40,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix='./data/jester/',
        ann_file='train.csv',
        pipeline=train_pipeline,
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        data_prefix='./data/jester/',
        ann_file='val.csv',
        pipeline=test_pipeline,
        test_mode=False
    ),
    test=dict(
        type=dataset_type,
        data_prefix='./data/jester/',
        ann_file='val.csv',
        pipeline=test_pipeline,
        test_mode=True
    ),
    test_gpu=dict(
        type=dataset_type,
        data_prefix='./data/jester/',
        ann_file='val.csv',
        pipeline=test_pipeline_gpu,
        test_mode=True
    ),
)

###

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealingLR', T_max=40,eta_min=0)
runner = dict(type='EpochBasedRunner', max_epochs=200)

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
