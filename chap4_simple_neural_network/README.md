### 第二次课程作业
**所有代码在`pytorch`框架下重写**
#### 问题2.1说明
1. 原本仓库中`tf2.0-exercise.ipynb`替换为`torch_exercise.ipynb`，用`pytorch`框架实现函数。
2. 原本仓库中`tutorial_mnist_fnn-numpy-exercise.ipynb`中梯度检验改用`pytorch`完成。同时，针对原始文件中梯度计算较为繁琐的问题，简化了`softmax`函数的梯度计算。对应文件夹中的文件`numpy_mnist.ipynb`。
3. 利用`pytorch`搭建全连接网络实现对于`MNIST`数据集的分类，对应文件`torch_mnist.ipynb`。
*注：所有结果已经保存在对应的`.ipynb`文件中*。

#### 问题2.2说明
1. 问题背景：利用`numpy`搭建两层全连接网络（ReLU激活），拟合$[-2\pi,2\pi]$上的$\sin x$函数。
2. 完成情况：
   1. 利用`numpy`实现线性层、ReLU激活函数、L2损失函数。
   2. 基于`numpy`实现静态计算图和其中节点梯度的反向传播。
3. 所有文件储存在`nn`文件夹下，其中
   1. `nn.py`储存计算图相关节点
   2. `base.py`储存基类型
   3. `model.py`储存网络模型
   4. `utils.py`储存数据集基类型，同时设计具体问题的数据集，支持动态可视化功能。
*注：静态计算图的遍历和梯度的传播参考了[UCB CS188 project](https://berkeleyai.github.io/cs188-website/)*
1. `configs`下存储了网络的拟合参数，包括超参数和精度要求。
2. 结果复现方法：在根目录`chap4_simple_neural_network`下
```bash
cd nn
python train.py
```
`report`文件夹下存储了问题报告，包括：函数定义、数据集分割、静态计算图实现、模型描述、拟合效果。