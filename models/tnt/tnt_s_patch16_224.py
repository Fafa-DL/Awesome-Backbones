# model settings
model_cfg = dict(
    backbone=dict(
        type='TNT',
        arch='s',
        img_size=224,
        patch_size=16,
        in_channels=3,
        ffn_ratio=4,
        qkv_bias=False,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        first_stride=4,
        num_fcs=2,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
        ]),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=5,
        in_channels=384,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)))

# dataloader pipeline
train_pipeline = (
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

)
val_pipeline = (
    dict(type='Resize', size=248),
    dict(type='CenterCrop', size=224),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

# train
data_cfg = dict(
    batch_size = 16,
    num_workers = 4,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = 'datas/tnt-small-p16_.pth',
        freeze_flag = False,
        freeze_layers = ('backbone',),
        epoches = 100,
    ),
    test=dict(
        ckpt = 'logs/MobileNetV3/2022-04-10-09-17-25/Train_Epoch098-Loss0.035.pth',
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options = dict(
            topk = (1,5),
            thrs = None,
            average_mode='none'
    )
    )
)

# batch 16
# lr = 1e-3 * 16 / 64
# optimizer
optimizer_cfg = dict(
    type='AdamW',
    lr=1e-3 * 16 / 64,
    weight_decay=0.05,)

# learning 
lr_config = dict(
    type='CosineAnnealingLrUpdater',
    min_lr=0,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=3,
    warmup_by_epoch=True
)
