# model settings
model_cfg = dict(
    backbone=dict(type='RegNet', arch='regnetx_6.4gf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1624,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# dataloader pipeline
# normalization params, in order of BGR
NORM_MEAN = [103.53, 116.28, 123.675]
NORM_STD = [57.375, 57.12, 58.395]

# lighting params, in order of RGB, from repo. pycls
EIGVAL = [0.2175, 0.0188, 0.0045]
EIGVEC = [[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.814],
          [-0.5836, -0.6948, 0.4203]]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Lighting',
        eigval=EIGVAL,
        eigvec=EIGVEC,
        alphastd=25.5,  # because the value range of images is [0,255]
        to_rgb=True
    ),  # BGR image from cv2 in LoadImageFromFile, convert to RGB here
    dict(type='Normalize', mean=NORM_MEAN, std=NORM_STD,
         to_rgb=True),  # RGB2BGR
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', mean=NORM_MEAN, std=NORM_STD, to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size = 32,
    num_workers = 4,
    train = dict(
        pretrained_flag = False,
        pretrained_weights = '',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = '',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=5e-5)

# learning 
lr_config = dict(
    type='CosineAnnealingLrUpdater',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.1,
    warmup_by_epoch=True
)
