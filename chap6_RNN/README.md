### 第三次课程作业
1. 完成`pytorch`版本的RNN，利用`../data/poems.txt`进行训练和测试。
2. `report`文件夹下存放实验训练过程和结果生成截图，其中：
   1. `early_training.png`记录训练起始阶段的输出。
   2. `late_training.png`记录训练中期的输出。
   3. `results.png`为利用开头词汇`日、红、山、夜、湖、海、月`生成诗歌的效果截图。
3. 对原有的代码进行了修改，包括：
   1. 主要利用`pad_sequence`和`pack_padded_sequence`让`LSTM`训练过程支持批量训练，原始代码中`batch_size`为1，训练效率较低。
   2. 分离训练和测试的部分代码，防止测试过程中计算梯度。原始代码中两部分一起放在`forward`中。
   3. 修改诗歌打印逻辑。
4. 结果复现方式
   ```bash
   python main.py
   ```
   同时训练好的模型记录在`data/rnn_lstm/models`中，可以直接加载权重使用。