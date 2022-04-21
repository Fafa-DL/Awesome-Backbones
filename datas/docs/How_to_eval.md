模型评估
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 确认`Awesome-Backbones/datas/annotations.txt`标签准备完毕
- 确认`Awesome-Backbones/datas/`下`test.txt`与`annotations.txt`对应
- 在`Awesome-Backbones/models/`下找到对应配置文件
- 按照`配置文件解释`修改参数，主要修改权重路径
- 在`Awesome-Backbones`打开终端运行
```
python tools/evaluation.py 'models/mobilenet/mobilenet_v3_small.py'
```


