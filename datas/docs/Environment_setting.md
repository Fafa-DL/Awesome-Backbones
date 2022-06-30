环境搭建
===========================

[![BILIBILI](https://raw.githubusercontent.com/Fafa-DL/readme-data/main/Bilibili.png)](https://space.bilibili.com/46880349)

- 建议使用[Anaconda](https://www.anaconda.com/)进行环境管理，创建环境命令如下
```bash
conda create -n [name] python=3.6 其中[name]改成自己的环境名，如[name]->torch，conda create -n torch python=3.6
```
- 我的测试环境如下
```
torch==1.7.1
torchvision==0.8.2
scipy==1.4.1
numpy==1.19.2
matplotlib==3.2.1
opencv_python==3.4.1.15
tqdm==4.62.3
Pillow==8.4.0
h5py==3.1.0
terminaltables==3.1.0
packaging==21.3
```
- 首先安装Pytorch。建议版本和我一致，进入[Pytorch](https://pytorch.org/)官网，点击` install previous versions of PyTorch`，以1.7.1为例，官网给出的安装如下，选择合适的cuda版本
```
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
- **安装完Pytorch后**，再运行
```bash
pip install -r requirements.txt
```