模型评估
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 确认`Awesome-Backbones/datas/annotations.txt`标签准备完毕
- 确认`Awesome-Backbones/datas/`下`test.txt`与`annotations.txt`对应
- 在`Awesome-Backbones/models/`下找到对应配置文件
- 按照`配置文件解释`修改参数，主要修改权重路径
- 在`Awesome-Backbones`打开终端运行
```
python tools/evaluation.py models/mobilenet/mobilenet_v3_small.py
```
- 单张图像测试，在`Awesome-Backbones`打开终端运行
```
python tools/single_test.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/mobilenet/mobilenet_v3_small.py
```
**参数说明**：

`img` : 被测试的单张图像路径

`config` : 模型配置文件，需注意修改配置文件中`data_cfg->test->ckpt`的权重路径，将使用该权重进行预测

`--device` : 推理所用设备，默认GPU

`--save-path` : 保存路径，默认不保存

![results](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/dandelion.jpg)

