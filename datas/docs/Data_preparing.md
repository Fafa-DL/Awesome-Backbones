制作数据集
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

## 1. 标签文件制作

- 本次演示以花卉数据集为例，目录结构如下：

```
├─flower_photos
│  ├─daisy
│  │      100080576_f52e8ee070_n.jpg
│  │      10140303196_b88d3d6cec.jpg
│  │      ...
│  ├─dandelion
│  │      10043234166_e6dd915111_n.jpg
│  │      10200780773_c6051a7d71_n.jpg
│  │      ...
│  ├─roses
│  │      10090824183_d02c613f10_m.jpg
│  │      102501987_3cdb8e5394_n.jpg
│  │      ...
│  ├─sunflowers
│  │      1008566138_6927679c8a.jpg
│  │      1022552002_2b93faf9e7_n.jpg
│  │      ...
│  └─tulips
│  │      100930342_92e8746431_n.jpg
│  │      10094729603_eeca3f2cb6.jpg
│  │      ...
```
- 在`Awesome-Backbones/datas/`中创建标签文件`annotations.txt`，按行将`类别名 索引`写入文件；
```
daisy 0
dandelion 1
roses 2
sunflowers 3
tulips 4
```
## 2. 数据集划分
- 打开`Awesome-Backbones/tools/split_data.py`
- 修改`原始数据集路径`以及`划分后的保存路径`，强烈建议划分后的保存路径`datasets`不要改动，在下一步都是默认基于文件夹进行操作
```
init_dataset = 'A:/flower_photos'
new_dataset = 'A:/Awesome-Backbones/datasets'
```
- 在`Awesome-Backbones/`下打开终端输入命令：
```
python tools/split_data.py
```
- 得到划分后的数据集格式如下：
```
├─...
├─datasets
│  ├─test
│  │  ├─daisy
│  │  ├─dandelion
│  │  ├─roses
│  │  ├─sunflowers
│  │  └─tulips
│  └─train
│      ├─daisy
│      ├─dandelion
│      ├─roses
│      ├─sunflowers
│      └─tulips
├─...
```
## 3. 数据集信息文件制作
- 确保划分后的数据集是在`Awesome-Backbones/datasets`下，若不在则在`get_annotation.py`下修改数据集路径；
```
datasets_path   = '你的数据集路径'
```
- 在`Awesome-Backbones/`下打开终端输入命令：
```
python tools/get_annotation.py
```
- 在`Awesome-Backbones/datas`下得到生成的数据集信息文件`train.txt`与`test.txt`
