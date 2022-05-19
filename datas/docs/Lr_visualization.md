学习率策略可视化
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 提供 `tools/vis_lr.py` 工具来可视化学习率。

**命令行**：

```bash
python tools/vis_lr.py \
    ${CONFIG_FILE} \
    [--dataset-size ${Dataset_Size}] \
    [--ngpus ${NUM_GPUs}] \
    [--save-path ${SAVE_PATH}] \
    [--title ${TITLE}] \
    [--style ${STYLE}] \
    [--window-size ${WINDOW_SIZE}] \
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--dataset-size` : 数据集的大小。如果指定，`datas/train.txt` 将被跳过并使用这个大小作为数据集大小，默认使用 `datas/train.txt` 所得数据集的大小。
- `--ngpus` : 使用 GPU 的数量。
- `--save-path` : 保存的可视化图片的路径，默认不保存。
- `--title` : 可视化图片的标题，默认为配置文件名。
- `--style` : 可视化图片的风格，默认为 `whitegrid`。
- `--window-size`: 可视化窗口大小，如果没有指定，默认为 `12*7`。如果需要指定，按照格式 `'W*H'`。

```{note}

部分数据集在解析标注阶段比较耗时，可直接将 `dataset-size` 指定数据集的大小，以节约时间。
```

**示例Step**：

```bash
python tools/vis_lr.py models/mobilenet/mobilenet_v3_small.py
```
![lr01](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/lr_mobilenet.png)

**示例Cos**：

```bash
python tools/vis_lr.py models/swin_transformer/base_224.py
```
![lr02](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/lr_swin.png)