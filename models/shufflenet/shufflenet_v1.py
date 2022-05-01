# model settings

model_cfg = dict(
    backbone=dict(type='ShuffleNetV1', groups=3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=960,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),))

# dataloader pipeline
train_pipeline = (
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='RandomErasing',p=0.2,ratio=(0.02,1/3)),
)
val_pipeline = (
    dict(type='Resize', size=256),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

# train
data_cfg = dict(
    batch_size = 32,
    num_workers = 4,
    train = dict(
        pretrained_flag = False,
        pretrained_weights = 'datas/mobilenet_v3_small.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = 'logs/20220202091725/Val_Epoch019-Loss0.215.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# batch 32
# lr = 0.5 * 32 / 1024
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.5 * 32 / 1024,
    momentum=0.9,
    weight_decay=0.00004)

# learning 
lr_config = dict(
    type='PolyLrUpdater',
    min_lr=0,
    by_epoch=False,
    warmup='constant',
    warmup_iters=5000, # 前5000步用warup更新学习率，可以自己根据数据集设置步数。
)
#lr_config = dict(type='StepLrUpdater', warmup='linear', warmup_iters=500, warmup_ratio=0.25,step=[30,60,90])

