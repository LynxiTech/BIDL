###

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SeqClif7Fc1DgItout',
        timestep=16, input_channels=2, h=128, w=128, nclass=11, cmode='spike', fmode='spike', amode='mean', noise=0,
        soma_params='all_share',
        neuron='lif', # neuron mode: 'lif' or 'lifplus'
        neuron_config=None # neron configs: 1.'lif': neuron_config=None; 2.'lifplus': neuron_config=[input_accum, rev_volt, fire_refrac, spike_init, trig_current, memb_decay], eg.[1,False,0,0,0,0]
    ),
    neck=None,
    head=dict(
        type='ClsHead',
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.3, loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True
    ))

###

dataset_type = 'DvsGesture'

train_pipeline = [
    dict(type='RandomCropVideo', size=128, padding=4),  # XXX
    dict(type='ToTensorType', keys=['img'], dtype='float32'),  # XXX
]
test_pipeline = [
    dict(type='ToTensorType', keys=['img'], dtype='float32'),  # XXX
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=0,  
    data_root='./data',
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(
        pipeline=test_pipeline,
    ),
    test=dict(
        pipeline=test_pipeline,
    ),
)


###

#optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001, nesterov=True)
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='MultiStepLR', milestones=[30, 60, 90])
#lr_config = dict(policy='MultiStepLR', milestones=[50, 90])
warmup=dict(warmup_ratio=1e-2, warmup_iters=100)
runner = dict(type='EpochBasedRunner', max_epochs=100)

###

checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

#fp16 = dict(loss_scale=512.0)  # half-percision
fp16 = None  # half-percision
