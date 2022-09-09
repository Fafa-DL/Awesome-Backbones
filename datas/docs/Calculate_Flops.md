获取模型Flops&Param
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)
- 提供 `tools/get_flops.py` 工具来计算模型参数量与浮点运算量。

**命令行**：

```bash
python tools/get_flops.py \
    ${CONFIG_FILE} \
    [--shape ${Shape}] \
```

**所有参数的说明**：

- `config` : 模型配置文件的路径。
- `--shape` : 输入图片尺寸，默认224


**示例Step**：

```bash
python tools/get_flops.py models/mobilenet/mobilenet_v3_small.py
```

**注意**：

- 官方给出的参数量与浮点运算量是基于ImageNet，也就是说默认分类数为`1000`，所以当你评估自己模型时请在配置文件中将`num_classes`修改为对应数量，因为将很大程度上影响结果
- 如果你有新增任何`基类`卷积/池化/采样功能，请在`utils/flops_counter.py/get_modules_mapping()`进行增加注册
