### 第二次课程作业
#### 问题2.3说明
1. 使用`pytorch`和`tensorflow`分别搭建CNN模型，完成MNIST数据集分类。对应文件为`CNN_pytorch.ipynb`和`CNN_tensorflow.ipynb`。针对新版`pytorch`和`tensorflow`中的一些特性，对原始文件做了相应修改。训练结果保存在文件中，两个版本的训练准确率都在96%以上，符合要求。
注：原始的`CNN_tensorflow.ipynb`适配`tensorflow=1.x`，对于现在广泛使用的`tensorflow=2.x`，其中定义`placeholders`和`sessions`的行为已不再适用，我们使用了`keras.Sequential`进行替换。