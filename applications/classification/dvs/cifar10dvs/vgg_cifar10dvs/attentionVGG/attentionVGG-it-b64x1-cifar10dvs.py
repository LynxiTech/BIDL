###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='AttentionVGGNet',
        timestep=10, input_channels=2, h=128, w=128, nclass=10, cmode='spike', noise=0
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

dataset_type = 'Cifar10Dvs'
train_pipeline = [
    dict(type='RandomCropVideo', size=128, padding=16),
    dict(type='RandomFlipVideo', flip_prob=0.5, direction='horizontal'),
    dict(type='ToTensorType', keys=['img'], dtype='float32'),
]
test_pipeline = [
    dict(type='ToTensorType', keys=['img'], dtype='float32'),
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    data_root='./data',
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    ))

###

# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer = dict(type='Adam', lr=0.01, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealingLR',T_max=32, eta_min=0)
warmup=dict(warmup_ratio=1e-1, warmup_iters=1000)
runner = dict(type='EpochBasedRunner', max_epochs=200)

###

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

fp16 = dict(loss_scale=512.0)  # half-percision
