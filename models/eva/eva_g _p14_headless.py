# Model settings
model_cfg = dict(
    backbone=dict(
        type='BEiT',
        arch='eva-g',
        img_size=224,
        patch_size=14,
        avg_token=True,
        layer_scale_init_value=0.0,
        output_cls_token=False,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
    ),
    neck=None,
    head=None
    )

# dataloader pipeline
img_norm_cfg = dict(
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255], std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

# train
data_cfg = dict(
    batch_size = 32,
    num_workers = 2,
    train = dict(
        pretrained_flag = True,
        pretrained_weights = 'beit-base_3rdparty_in1k_20221114-c0a4df23.pth',
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
# lr = 0.01 *32 /64
# optimizer
optimizer_cfg = dict(
    type='AdamW',
    lr=0.001 * 32 / 64,
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
    warmup_iters=20,
    warmup_by_epoch=True)