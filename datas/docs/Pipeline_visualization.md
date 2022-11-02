查看数据增强结果
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)
- 提供 `tools/vis_pipeline.py` 工具来批量显示数据增强pipeline结果

**命令行**：

```bash
python tools/vis_pipeline.py \
    ${input} \
    ${CONFIG_FILE} \
    [--skip-type ${SKIP-TYPE}] \
    [--output-dir ${OUTPUT-DIR}] \
    [--phase ${PHASE}] \
    [--number ${NUMBER}] \
    [--sleep ${SLEEP}] \
    [--show ${SHOW}] \
```

**所有参数的说明**：

- `input`: 待显示的图片文件夹路径；
- `config` : 模型配置文件路径；
- `--skip-type` : 跳过pipeline中的某些增强方式，默认['ToTensor', 'Normalize', 'ImageToTensor', 'Collect']；
- `--output-dir`: 保存增强后的图片文件夹路径，默认不保存；
- `--phase`: 指定显示哪个阶段的pipeline，默认train，支持['train', 'test', 'val']；
- `--number`: 指定显示/保存的图片数，默认全部显示；
- `--sleep`: 每张图片显示时间，默认1秒；
- `--show`: 是否显示增强后的图片，默认不显示。


**示例Step**：

```bash
tools/vis_pipeline.py datasets/test/dandelion models/swin_transformer/small_224.py --show --number 10 --sleep 0.5 --output-dir aug_results
```

**注意**：

- 很有必要预览整个数据集的数据增强结果，如果遮挡/变形失真的图片比例`占据较多`，很有可能导致准确率下降，因为其主导了训练loss走向，建议选择合适的增强方式，配置文件只是默认!

![](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/fail01.jpg) ![](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/fail02.jpg) ![](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/fail03.jpg)
