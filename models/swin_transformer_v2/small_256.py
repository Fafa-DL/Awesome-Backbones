# model settings
model_cfg = dict(
    backbone=dict(
        type='SwinTransformerV2',
        arch='small',
        img_size=256,
        drop_path_rate=0.3),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=768,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ])

# dataloader pipeline
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
rand_increasing_policies = [
    dict(type='AutoContrast'),
    dict(type='Equalize'),
    dict(type='Invert'),
    dict(type='Rotate', magnitude_key='angle', magnitude_range=(0, 30)),
    dict(type='Posterize', magnitude_key='bits', magnitude_range=(4, 0)),
    dict(type='Solarize', magnitude_key='thr', magnitude_range=(256, 0)),
    dict(
        type='SolarizeAdd',
        magnitude_key='magnitude',
        magnitude_range=(0, 110)),
    dict(
        type='ColorTransform',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(type='Contrast', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Brightness', magnitude_key='magnitude',
        magnitude_range=(0, 0.9)),
    dict(
        type='Sharpness', magnitude_key='magnitude', magnitude_range=(0, 0.9)),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='horizontal'),
    dict(
        type='Shear',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.3),
        direction='vertical'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='horizontal'),
    dict(
        type='Translate',
        magnitude_key='magnitude',
        magnitude_range=(0, 0.45),
        direction='vertical')
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        size=256,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies=rand_increasing_policies,
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in img_norm_cfg['mean'][::-1]],
            interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=img_norm_cfg['mean'][::-1],
        fill_std=img_norm_cfg['std'][::-1]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        size=(292, -1),  # ( 256 / 224 * 256 )
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
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

# batch 32
# lr = 5e-4 * 16 / 64
# optimizer
optimizer_cfg = dict(
    type='AdamW',
    lr=5e-4 * 16 / 64,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),)

# learning 
lr_config = dict(
    type='CosineAnnealingLrUpdater',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=3,
    warmup_by_epoch=True
)
