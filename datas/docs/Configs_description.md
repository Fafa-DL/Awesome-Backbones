配置文件解释
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 每个模型均对应有各自的配置文件，保存在`Awesome-Backbones/models`下
- Model
```python
'''
由backbone、neck、head、head.loss构成一个完整模型；

type与相应结构对应，其后紧接搭建该结构所需的参数，每个配置文件均已设置好；

需修改的地方：num_classes修改为对应数量，如花卉数据集为五类，则num_classes=5
'''
model_cfg = dict(
    backbone=dict(type='MobileNetV3', arch='large'), 
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='StackedLinearClsHead',
        num_classes=1000,
        in_channels=960,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='HSwish'),
        loss=dict(
            type='CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(
            type='Normal', layer='Linear', mean=0., std=0.01, bias=0.),
        topk=(1, 5)))
```
- Datasets
```python
'''
该部分对应构建训练/测试时的Datasets，使用torchvision.transforms进行预处理；

size=224为最终处理后，喂入网络的图像尺寸；

Normalize对应归一化，默认使用ImageNet数据集均值与方差，若你有自己数据集的参数，可以选择覆盖。
'''
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
```
- Train/Test
```python
'''
该部分对应训练/测试所需参数；

batch_size                 : 根据自己设备进行调整，建议为`2`的倍数
num_workers            : Dataloader中加载数据的线程数，根据自己设备调整
pretrained_flag          : 若使用预训练权重，则设置为True
pretrained_weights    : 权重路径
freeze_flag                : 若冻结某部分训练，则设置为True
freeze_layers              :可选冻结的有backbone, neck, head
epoches                    : 最大迭代周期

ckpt : 评估模型所需的权重文件
`其余参数均不用改动`
'''
data_cfg = dict(
    batch_size = 32,
    num_workers = 4,
    train = dict(
        pretrained_flag = False,
        pretrained_weights = './datas/mobilenet_v3_small.pth',
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
    ))
)
```
- Optimizer
```python
'''
训练时的优化器，与torch.optim对应

type : 'RMSprop'对应torch.optim.RMSprop，可在torch.optim查看
PyTorch支持Adadelta、Adagrad、Adam、AdamW、SparseAdam、Adamax、ASGD、SGD、Rprop、RMSprop、Optimizer、LBFGS
可以根据自己需求选择优化器

lr      : 初始学习率，可根据自己Batch Size调整
ckpt : 评估模型所需的权重文件

其余参数均不用改动
'''
optimizer_cfg = dict(
    type='RMSprop',
    lr=0.001,
    alpha=0.9,
    momentum=0.9,
    eps=0.0316,
    weight_decay=1e-5)
```
- Learning Rate
```python
'''
学习率更新策略，各方法可在Awesome-Backbones/core/optimizers/lr_update.py查看

StepLrUpdater                    : 线性递减
CosineAnnealingLrUpdater : 余弦退火

by_epoch                : 是否每个Epoch更新学习率
warmup                  : 在正式使用学习率更新策略前先用warmup小学习率训练，可选constant, linear, exp
warmup_ratio          : 与`Optimizer`中的`lr`结合所选warmup方式进行学习率运算更新
warmup_by_epoch  : 作用与`by_epoch`类似，若为False，则为每一步（Batch）进行更新，否则每周期
warmup_iters          : warmup作用时长，warmup_by_epoch为True则代表周期，False则代表步数
'''
lr_config = dict(
    type='CosineAnnealingLrUpdater',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=20,
    warmup_by_epoch=True)
```
