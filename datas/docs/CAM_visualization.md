类别激活图可视化
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 提供 `tools/vis_cam.py` 工具来可视化类别激活图。请使用 `pip install grad-cam` 安装依赖，版本≥1.3.6
- 在`Awesome-Backbones/models/`下找到对应配置文件
- 修改data_cfg中test的ckpt路径，改为训练完毕的权重
目前支持的方法有：

| Method     | What it does |
|:----------:|:------------:|
| GradCAM    | 使用平均梯度对 2D 激活进行加权 |
| GradCAM++  | 类似 GradCAM，但使用了二阶梯度 |
| XGradCAM   | 类似 GradCAM，但通过归一化的激活对梯度进行了加权 |
| EigenCAM   | 使用 2D 激活的第一主成分（无法区分类别，但效果似乎不错）|
| EigenGradCAM  | 类似 EigenCAM，但支持类别区分，使用了激活 \* 梯度的第一主成分，看起来和 GradCAM 差不多，但是更干净 |
| LayerCAM  | 使用正梯度对激活进行空间加权，对于浅层有更好的效果 |

![CAM02](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/cam02.png)

**命令行**：

```bash
python tools/vis_cam.py \
    ${IMG} \
    ${CONFIG_FILE} \
    [--target-layers ${TARGET-LAYERS}] \
    [--preview-model] \
    [--method ${METHOD}] \
    [--target-category ${TARGET-CATEGORY}] \
    [--save-path ${SAVE_PATH}] \
    [--vit-like] \
    [--num-extra-tokens ${NUM-EXTRA-TOKENS}]
    [--aug_smooth] \
    [--eigen_smooth] \
    [--device ${DEVICE}] \
```

**所有参数的说明**：

- `img`：目标图片路径。
- `config`：模型配置文件的路径。需注意修改配置文件中`data_cfg->test->ckpt`的权重路径，将使用该权重进行预测
- `--target-layers`：所查看的网络层名称，可输入一个或者多个网络层, 如果不设置，将使用最后一个`block`中的`norm`层。
- `--preview-model`：是否查看模型所有网络层。
- `--method`：类别激活图图可视化的方法，目前支持 `GradCAM`, `GradCAM++`, `XGradCAM`, `EigenCAM`, `EigenGradCAM`, `LayerCAM`，不区分大小写。如果不设置，默认为 `GradCAM`。
- `--target-category`：查看的目标类别，如果不设置，使用模型检测出来的类别做为目标类别。
- `--save-path`：保存的可视化图片的路径，默认不保存。
- `--eigen-smooth`：是否使用主成分降低噪音，默认不开启。
- `--vit-like`: 是否为 `ViT` 类似的 Transformer-based 网络
- `--num-extra-tokens`: `ViT` 类网络的额外的 tokens 通道数，默认使用主干网络的 `num_extra_tokens`。
- `--aug-smooth`：是否使用测试时增强
- `--device`：使用的计算设备，如果不设置，默认为'cpu'。

```{note}
在指定 `--target-layers` 时，如果不知道模型有哪些网络层，可使用命令行添加 `--preview-model` 查看所有网络层名称；
```
**示例（CNN）**：
1. 使用不同方法可视化 `MobileNetV3`，默认 `target-category` 为模型检测的结果，使用默认推导的 `target-layers`。
```
python tools/vis_cam.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/mobilenet/mobilenet_v3_small.py
```
2. 指定同一张图中不同类别的激活图效果图，给定类别索引即可
```
python tools/vis_cam.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/mobilenet/mobilenet_v3_small.py --target-category 1
```
3. 使用 `--eigen-smooth` 以及 `--aug-smooth` 获取更好的可视化效果。
```
python tools/vis_cam.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/mobilenet/mobilenet_v3_small.py --eigen-smooth --aug-smooth
```
**示例（Transformer）**：

对于 Transformer-based 的网络，比如 ViT、T2T-ViT 和 Swin-Transformer，特征是被展平的。为了绘制 CAM 图，需要指定 `--vit-like` 选项，从而让被展平的特征恢复方形的特征图。

除了特征被展平之外，一些类 ViT 的网络还会添加额外的 tokens。比如 ViT 和 T2T-ViT 中添加了分类 token，DeiT 中还添加了蒸馏 token。在这些网络中，分类计算在最后一个注意力模块之后就已经完成了，分类得分也只和这些额外的 tokens 有关，与特征图无关，也就是说，分类得分对这些特征图的导数为 0。因此，我们不能使用最后一个注意力模块的输出作为 CAM 绘制的目标层。

另外，为了去除这些额外的 toekns 以获得特征图，我们需要知道这些额外 tokens 的数量。MMClassification 中几乎所有 Transformer-based 的网络都拥有 `num_extra_tokens` 属性。而如果你希望将此工具应用于新的，或者第三方的网络，而且该网络没有指定 `num_extra_tokens` 属性，那么可以使用 `--num-extra-tokens` 参数手动指定其数量。
1. 对 `Swin Transformer` 使用默认 `target-layers` 进行 CAM 可视化：
```
python tools/vis_cam.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/swin_transformer/tiny_224.py --vit-like
```
2. 对 `Vision Transformer(ViT)` 进行 CAM 可视化（经测试其实不加--target-layer即默认效果也差不多）：
```
python tools/vis_cam.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/vision_transformer/vit_base_p16_224.py --vit-like --target-layers backbone.layers[-1].ln1
```
3. 对 `T2T-ViT` 进行 CAM 可视化：
```
python tools/vis_cam.py datasets/test/dandelion/14283011_3e7452c5b2_n.jpg models/t2t_vit/t2t_vit_t_14.py --vit-like --target-layers backbone.encoder[-1].ln1
```

![CAM01](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/backbones/cam01.jpg)