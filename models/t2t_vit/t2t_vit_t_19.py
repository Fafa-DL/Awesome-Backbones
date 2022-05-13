# model settings
embed_dims = 448

model_cfg = dict(
    backbone=dict(
        type='T2T_ViT',
        img_size=224,
        in_channels=3,
        embed_dims=embed_dims,
        t2t_cfg=dict(
            token_dims=64,
            use_performer=False,
        ),
        num_layers=19,
        layer_cfgs=dict(
            num_heads=7,
            feedforward_channels=3 * embed_dims,  # mlp_ratio = 3
        ),
        drop_path_rate=0.1,
        init_cfg=[
            dict(type='TruncNormal', layer='Linear', std=.02),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        ]),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=embed_dims,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            mode='original',
        ),
        topk=(1, 5),
        init_cfg=dict(type='TruncNormal', layer='Linear', std=.02)),
    )

# dataloader pipeline
train_pipeline = (
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
# lr = 5e-4 * 32 / 64
# optimizer
optimizer_cfg = dict(
    type='AdamW',
    lr=5e-4 * 32 / 64,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),)

# learning 
lr_config = dict(
    type='CosineAnnealingCooldownLrUpdater',
    min_lr=1e-5,
    cool_down_time=10,
    cool_down_ratio=0.1,
    by_epoch=True,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-6)