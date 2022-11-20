训练自己数据集
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 确认`Awesome-Backbones/datas/annotations.txt`标签准备完毕
- 确认`Awesome-Backbones/datas/`下`train.txt`与`test.txt`与`annotations.txt`对应
- 选择想要训练的模型，在`Awesome-Backbones/models/`下找到对应配置文件
- 按照`配置文件解释`修改参数
- 在`Awesome-Backbones`打开终端运行
```
python tools/train.py models/mobilenet/mobilenet_v3_small.py
```
**命令行**：

```bash
python tools/train.py \
    ${CONFIG_FILE} \
    [--resume-from] \
    [--seed] \
    [--device] \
    [--gpu-id] \
    [--split-validation] \
    [--ratio] \
    [--deterministic] \
```

**所有参数的说明**：

- `config`：模型配置文件的路径。
- `--resume-from`：从中断处恢复训练，提供权重路径，`务必注意正确的恢复方式是从Last_Epoch***.pth`，如--resume-from logs/SwinTransformer/2022-02-08-08-27-41/Last_Epoch15.pth
- `--seed`：设置随机数种子，默认按照环境设置
- `--device`：设置GPU或CPU训练
- `--gpu-id`：指定GPU设备，默认为0（单卡基本均为0不用改动）
- `--split-validation`：是否从训练集中划分验证集，划分比例默认0.2，否则直接将测试集用于验证
- `--ratio`：从训练集中划分验证集的比例，默认0.2，且shuffle后随机从训练集某fold挑选
- `--deterministic`：多GPU训练相关，暂不用设置