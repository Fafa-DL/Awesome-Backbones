Awesome backbones for classification
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

## 更新日志

**`2022.04.19`** : 创建仓库
**`2022.05.01`** : 更新Swin Transformer

## 测试环境

- Numpy       1.16.0
- PyTorch      1.7.1
- TorchVision 0.8.2
- tqdm          4.62.3 
- Pillow         8.4.0

## 资料
|数据集|视频教程|人工智能技术探讨群|
|---|---|---|
|[`花卉数据集` 提取码：0zat](https://pan.baidu.com/s/1137y4l-J3AgyCiC_cXqIqw)|[点我跳转](https://www.bilibili.com/video/BV1SY411P7Nd)|[1群：78174903](https://jq.qq.com/?_wv=1027&k=lY5KVICA)<br/>[2群：571218304](https://jq.qq.com/?_wv=1027&k=ZCDCT3xV)<br/>[3群：584723646](https://jq.qq.com/?_wv=1027&k=bakez5Yz)


## 教程
- [数据集准备](https://github.com/Fafa-DL/Awesome-Backbones/blob/main/datas/docs/Data_preparing.md)
- [配置文件解释](https://github.com/Fafa-DL/Awesome-Backbones/blob/main/datas/docs/Configs_description.md)
- [训练](https://github.com/Fafa-DL/Awesome-Backbones/blob/main/datas/docs/How_to_train.md)
- [模型评估](https://github.com/Fafa-DL/Awesome-Backbones/blob/main/datas/docs/How_to_eval.md)

## 模型
- [x] [LeNet5](https://ieeexplore.ieee.org/document/6795724)
- [x] [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [x] [VGG](https://arxiv.org/abs/1409.1556)
- [x] [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- [x] [ResNeXt](https://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html)
- [x] [SEResNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)
- [x] [SEResNeXt](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)
- [x] [RegNet](https://arxiv.org/abs/2003.13678)
- [x] [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [x] [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [x] [ShuffleNetV1](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html)
- [x] [ShuffleNetV2](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf)
- [x] [EfficientNet](https://arxiv.org/abs/1905.11946v5)
- [x] [RepVGG](https://arxiv.org/abs/2101.03697)
- [x] [Res2Net](https://arxiv.org/pdf/1904.01169.pdf)
- [x] [ConvNeXt](https://arxiv.org/abs/2201.03545v1)
- [x] [HRNet](https://arxiv.org/abs/1908.07919v2)
- [x] [ConvMixer](https://arxiv.org/abs/2201.09792)
- [x] [CSPNet](https://arxiv.org/abs/1911.11929)
- [x] [Swin-Transformer](https://arxiv.org/pdf/2103.14030.pdf)
- [ ] [Vision-Transformer](https://arxiv.org/pdf/2010.11929.pdf)
- [ ] [Transformer-in-Transformer](https://arxiv.org/pdf/2103.00112.pdf)
- [ ] [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf)
- [ ] [DeiT](https://arxiv.org/pdf/2012.12877.pdf)
- [ ] [Conformer](https://arxiv.org/pdf/2105.03889.pdf)
- [ ] [T2T-ViT](https://arxiv.org/pdf/2101.11986.pdf)
- [ ] [Twins](http://arxiv-export-lb.library.cornell.edu/pdf/2104.13840)
- [ ] [PoolFormer](https://arxiv.org/pdf/2111.11418.pdf)

## 预训练权重

| 名称 | 权重 | 名称 | 权重 | 名称 | 权重 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| **LeNet5** | None | **AlexNet** | None | **VGG** | [VGG-11](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_batch256_imagenet_20210208-4271cd6c.pth)<br/>[VGG-13](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_batch256_imagenet_20210208-4d1d6080.pth)<br/>[VGG-16](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth)<br/>[VGG-19](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.pth)<br/>[VGG-11-BN](https://download.openmmlab.com/mmclassification/v0/vgg/vgg11_bn_batch256_imagenet_20210207-f244902c.pth)<br/>[VGG-13-BN](https://download.openmmlab.com/mmclassification/v0/vgg/vgg13_bn_batch256_imagenet_20210207-1a8b7864.pth)<br/>[VGG-16-BN](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_bn_batch256_imagenet_20210208-7e55cd29.pth)<br/>[VGG-19-BN](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_bn_batch256_imagenet_20210208-da620c4f.pth)|
| **ResNet** |[ResNet-18](https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth)<br/>[ResNet-34](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth)<br/>[ResNet-50](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth)<br/>[ResNet-101](https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth)<br/>[ResNet-152](https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_8xb32_in1k_20210901-4d7582fa.pth) | **ResNetV1C** | [ResNetV1C-50](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c50_8xb32_in1k_20220214-3343eccd.pth)<br/>[ResNetV1C-101](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c101_8xb32_in1k_20220214-434fe45f.pth)<br/>[ResNetV1C-152](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1c152_8xb32_in1k_20220214-c013291f.pth) |**ResNetV1D** | [ResNetV1D-50](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d50_b32x8_imagenet_20210531-db14775a.pth)<br/>[ResNetV1D-101](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth)<br/>[ResNetV1D-152](https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d152_b32x8_imagenet_20210531-278cf22a.pth) |
| **ResNeXt** | [ResNeXt-50](https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth)<br/>[ResNeXt-101](https://download.openmmlab.com/mmclassification/v0/resnext/resnext101_32x4d_b32x8_imagenet_20210506-e0fa3dd5.pth)<br/>[ResNeXt-152](https://download.openmmlab.com/mmclassification/v0/resnext/resnext152_32x4d_b32x8_imagenet_20210524-927787be.pth) | **SEResNet** | [SEResNet-50](https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet50_batch256_imagenet_20200804-ae206104.pth)<br/>[SEResNet-101](https://download.openmmlab.com/mmclassification/v0/se-resnet/se-resnet101_batch256_imagenet_20200804-ba5b51d4.pth)| **SEResNeXt**| None|
| **RegNet** |[RegNetX-400MF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-400mf_8xb128_in1k_20211213-89bfc226.pth)<br/>[RegNetX-800MF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-800mf_8xb128_in1k_20211213-222b0f11.pth)<br/>[RegNetX-1.6GF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-1.6gf_8xb128_in1k_20211213-d1b89758.pth)<br/>[RegNetX-3.2GF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-3.2gf_8xb64_in1k_20211213-1fdd82ae.pth)<br/>[RegNetX-4.0GF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-4.0gf_8xb64_in1k_20211213-efed675c.pth)<br/>[RegNetX-6.4GF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-6.4gf_8xb64_in1k_20211215-5c6089da.pth)<br/>[RegNetX-8.0GF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-8.0gf_8xb64_in1k_20211213-9a9fcc76.pth)<br/>[RegNetX-12GF](https://download.openmmlab.com/mmclassification/v0/regnet/regnetx-12gf_8xb64_in1k_20211213-5df8c2f8.pth) | **MobileNetV2** | [MobileNetV2](https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth) |**MobileNetV3** | [MobileNetV3-Small](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_small-8427ecf0.pth)<br/>[MobileNetV3-Large](https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/convert/mobilenet_v3_large-3ea3c186.pth) |
| **ShuffleNetV1** |[ShuffleNetV1](https://download.openmmlab.com/mmclassification/v0/shufflenet_v1/shufflenet_v1_batch1024_imagenet_20200804-5d6cec73.pth) | **ShuffleNetV2** | [ShuffleNetV2](https://download.openmmlab.com/mmclassification/v0/shufflenet_v2/shufflenet_v2_batch1024_imagenet_20200812-5bf4721e.pth) |**EfficientNet** | [EfficientNet-B0](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth)<br/>[EfficientNet-B1](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth)<br/>[EfficientNet-B2](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b2_3rdparty_8xb32_in1k_20220119-ea374a30.pth)<br/>[EfficientNet-B3](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32_in1k_20220119-4b4d7487.pth)<br/>[EfficientNet-B4](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32_in1k_20220119-81fd4077.pth)<br/>[EfficientNet-B5](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b5_3rdparty_8xb32_in1k_20220119-e9814430.pth)<br/>[EfficientNet-B6](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b6_3rdparty_8xb32-aa_in1k_20220119-45b03310.pth)<br/>[EfficientNet-B7](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth)<br/>[EfficientNet-B8](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b8_3rdparty_8xb32-aa-advprop_in1k_20220119-297ce1b7.pth) |
| **RepVGG** |[RepVGG-A0](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_3rdparty_4xb64-coslr-120e_in1k_20210909-883ab98c.pth)<br/>[RepVGG-A1](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_3rdparty_4xb64-coslr-120e_in1k_20210909-24003a24.pth) <br/>[RepVGG-A2](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A2_3rdparty_4xb64-coslr-120e_in1k_20210909-97d7695a.pth)<br/>[RepVGG-B0](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B0_3rdparty_4xb64-coslr-120e_in1k_20210909-446375f4.pth)<br/>[RepVGG-B1](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1_3rdparty_4xb64-coslr-120e_in1k_20210909-750cdf67.pth)<br/>[RepVGG-A1](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A1_3rdparty_4xb64-coslr-120e_in1k_20210909-24003a24.pth)<br/>[RepVGG-B1g2](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g2_3rdparty_4xb64-coslr-120e_in1k_20210909-344f6422.pth)<br/>[RepVGG-B1g4](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B1g4_3rdparty_4xb64-coslr-120e_in1k_20210909-d4c1a642.pth)<br/>[RepVGG-B2](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2_3rdparty_4xb64-coslr-120e_in1k_20210909-bd6b937c.pth)<br/>[RepVGG-B2g4](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-7b7955f0.pth)<br/>[RepVGG-B2g4](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B2g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-7b7955f0.pth)<br/>[RepVGG-B3](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-dda968bf.pth)<br/>[RepVGG-B3g4](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-B3g4_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-4e54846a.pth)<br/>[RepVGG-D2se](https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-D2se_3rdparty_4xb64-autoaug-lbs-mixup-coslr-200e_in1k_20210909-cf3139b7.pth)| **Res2Net** | [Res2Net-50-14w-8s](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w14-s8_3rdparty_8xb32_in1k_20210927-bc967bf1.pth)<br/>[Res2Net-50-26w-8s](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth)<br/>[Res2Net-101-26w-4s](https://download.openmmlab.com/mmclassification/v0/res2net/res2net101-w26-s4_3rdparty_8xb32_in1k_20210927-870b6c36.pth)<br/> |**ConvNeXt** | [ConvNeXt-Tiny](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth)<br/>[ConvNeXt-Small](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_3rdparty_32xb128_in1k_20220124-d39b5192.pth)<br/>[ConvNeXt-Base](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_in21k-pre-3rdparty_32xb128_in1k_20220124-eb2d6ada.pth)<br/>[ConvNeXt-Large](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_in21k-pre-3rdparty_64xb64_in1k_20220124-2412403d.pth)<br/>[ConvNeXt-XLarge](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-xlarge_in21k-pre-3rdparty_64xb64_in1k_20220124-76b6863d.pth) |
| **HRNet** |[HRNet-W18](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w18_3rdparty_8xb32_in1k_20220120-0c10b180.pth)<br/>[HRNet-W30](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w30_3rdparty_8xb32_in1k_20220120-8aa3832f.pth) <br/>[HRNet-W32](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w32_3rdparty_8xb32_in1k_20220120-c394f1ab.pth)<br/>[HRNet-W40](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w40_3rdparty_8xb32_in1k_20220120-9a2dbfc5.pth)<br/>[HRNet-W44](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w44_3rdparty_8xb32_in1k_20220120-35d07f73.pth)<br/>[HRNet-W48](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w48_3rdparty_8xb32_in1k_20220120-e555ef50.pth)<br/>[HRNet-W64](https://download.openmmlab.com/mmclassification/v0/hrnet/hrnet-w64_3rdparty_8xb32_in1k_20220120-19126642.pth) | **ConvMixer** | [ConvMixer-768/32](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-768-32_3rdparty_10xb64_in1k_20220323-bca1f7b8.pth)<br/>[ConvMixer-1024/20](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-1024-20_3rdparty_10xb64_in1k_20220323-48f8aeba.pth)<br/>[ConvMixer-1536/20](https://download.openmmlab.com/mmclassification/v0/convmixer/convmixer-1536_20_3rdparty_10xb64_in1k_20220323-ea5786f3.pth) |**CSPNet** | [CSPDarkNet50](https://download.openmmlab.com/mmclassification/v0/cspnet/cspdarknet50_3rdparty_8xb32_in1k_20220329-bd275287.pth)<br/>[CSPResNet50](https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnet50_3rdparty_8xb32_in1k_20220329-dd6dddfb.pth)<br/>[CSPResNeXt50](https://download.openmmlab.com/mmclassification/v0/cspnet/cspresnext50_3rdparty_8xb32_in1k_20220329-2cc84d21.pth) |
|**Swin Transformer**|[tiny-224](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth)<br/>[small-224](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth)<br/>[base-224](https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth)<br/>[large-224](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window7_224_22kto1k-5f0996db.pth)<br/>[base-384](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_base_patch4_window12_384_22kto1k-d59b0d1d.pth)<br/>[large-384](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_large_patch4_window12_384_22kto1k-0a40944b.pth)

## 参考
```
@repo{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```