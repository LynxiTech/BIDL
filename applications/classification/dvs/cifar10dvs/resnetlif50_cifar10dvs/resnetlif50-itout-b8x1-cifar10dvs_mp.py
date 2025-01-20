###

models_compile_inputshape = [[1, 2, 128, 128], [1, 512, 16, 16]] # dim 0 represents the number of model slice; dim 1 represent the input shape of model slice;
model_0 = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetLifItout_MP',
        timestep=10,
        depth=50, nclass=10,
        down_t=[1, 'avg'],
        input_channels=2,
        noise=1e-5,
        soma_params='all_share',
        cmode='spike',
        split=[0, 6] # layers included of model_0
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1, loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

model_1 = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNetLifItout_MP',
        timestep=10,
        depth=50, nclass=10,
        down_t=[1, 'avg'],
        input_channels=2,
        noise=1e-5,
        soma_params='all_share',
        cmode='spike',
        split=[6, 16] # layers included of model_1
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
    dict(type='ToTensorType', keys=[0], dtype='float32'),
]
test_pipeline = [
    dict(type='ToTensorType', keys=[0], dtype='float32'),
]
data = dict(
    samples_per_gpu=8,
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


optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing',T_max=40, min_lr=0)
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

# dim 0 represents the number of timesteps slices;
# dim 1 represents the number of model segments;
# value represents the device id;
# eg. lynxi_devices = [[0,1],[2,3],[4,5]], timesteps slices are 3, model segments are 2, device ids are 0,1,2,3,4,5.
# lynxi_devices = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19],[20,21],[22,23]]
lynxi_devices = [[0,1]]