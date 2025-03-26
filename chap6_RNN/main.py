import os
import logging
import sys
import numpy as np
import collections
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

import rnn

start_token = "G"
end_token = "E"
batch_size = 64


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector have two dimensions, first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(
        file_name,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(":")
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(" ", "")
                if (
                    "_" in content
                    or "(" in content
                    or "（" in content
                    or "《" in content
                    or "[" in content
                    or start_token in content
                    or end_token in content
                ):
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[: len(words)] + (" ",)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    # ! convert each poem to one line, and convert words to number encoding, shape for poems_vector: (#poems, #words (varying))
    # ! words consists of all words in the dataset
    # ! word_init_map is a dict mapping from words to number encoding
    return poems_vector, word_int_map, words


def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(
        file_name,
        "r",
        encoding="utf-8",
    ) as f:
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(" " " ", "").replace("，", "").replace("。", "")
                    if (
                        "_" in content
                        or "(" in content
                        or "（" in content
                        or "《" in content
                        or "[" in content
                        or start_token in content
                        or end_token in content
                    ):
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[: len(words)] + (" ",)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def run_training(data_dir, resume: bool = False):
    # 处理数据集
    # poems as training data, tangshi as test data
    poems_vector, word_to_int, _ = process_poems1(os.path.join(data_dir, "poems.txt"))
    # 生成batch
    print("finish loading data")
    BATCH_SIZE = 100
    label_pad_value = -100
    vocab_size = len(word_to_int) + 1
    total_epoch = 10

    torch.manual_seed(5)
    word_embedding = rnn.word_embedding(vocab_length=vocab_size, embedding_dim=100)
    rnn_model = rnn.RNN_model(
        batch_sz=BATCH_SIZE,
        vocab_len=vocab_size,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    )

    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)
    # optimizer = optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss(ignore_index=label_pad_value)
    if resume:
        rnn_model.load_state_dict(torch.load(os.path.join(data_dir, "models/poem_generator_rnn.pth")))

    for epoch in range(total_epoch):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        # data shape: (#chunks, batch, seq_length (varying))
        # y is x shift right one step
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch]  # (batch , time_step)

            # ! pad the sequence and pack in LSTM
            data_tensors_y = [torch.tensor(data, dtype=torch.long) for data in batch_y]
            padded_data_y = pad_sequence(data_tensors_y, batch_first=True, padding_value=label_pad_value).view(-1)
            
            pre = rnn_model(batch_x) # ret shape (batch * max_seq_length, vocab_size)
            loss = loss_fun(pre, padded_data_y) # loss function with padded entries

            # logging info
            if batch % 100 == 0 and batch:
                sample_x = pre.view(BATCH_SIZE, -1, vocab_size)[0]
                sample_y = batch_y[0]
                _, pre = torch.max(sample_x, dim=1)
                print(
                    "prediction", pre.data.tolist()
                )  # the following  three line can print the output and the prediction
                print(
                    "b_y       ", sample_y
                )  # And you need to take a screenshot and then past is to your homework paper.
                print("*" * 30)
                logging.info(f"epoch: {epoch}, loss: {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

        if epoch % 3 == 0 and epoch:
            torch.save(rnn_model.state_dict(), os.path.join(data_dir, "models/poem_generator_rnn.pth"))
            print("finish saving model")


def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    poem_sentences = poem.split("。")
    for s in poem_sentences:
        if s != "" and s != end_token:
            print(s + "。")

def prepare_test(data_dir, ):
    _, word_int_map, vocabularies = process_poems1(os.path.join(data_dir, "poems.txt"))
    word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=100)
    rnn_model = rnn.RNN_model(
        batch_sz=64,
        vocab_len=len(word_int_map) + 1,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    )

    rnn_model.load_state_dict(torch.load(os.path.join(data_dir, "models/poem_generator_rnn.pth")))
    return word_int_map, vocabularies, rnn_model

def gen_poem(begin_word, word_int_map, vocabularies, rnn_model):
    # 指定开始的字
    poem = begin_word
    word = begin_word
    while word != end_token:
        # print(poem)
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64) # (seq_length, )
        output = rnn_model.predict(input)
        word = to_word(output.data.flatten(), vocabularies)
        poem += word
        if len(poem) > 50:
            break
    return poem


if __name__ == "__main__":
    data_dir = "../data/rnn_lstm"
    log_dir = "../logs"
    os.makedirs(os.path.join(data_dir, "models"), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # logging configurations
    filehandler = logging.FileHandler(os.path.join(log_dir, "rnn_lstm.log"), mode='a')
    stdhandler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[filehandler, stdhandler])
    run_training(data_dir=data_dir, resume=True)

    word_int_map, vocabularies, rnn_model = prepare_test(data_dir)
    for begin_word in [
        "日",
        "红",
        "山",
        "夜",
        "湖",
        "海",
        "月"
    ]:
        print('-' * 30)
        # gen_poem(begin_word, word_int_map, vocabularies, rnn_model)
        pretty_print_poem(gen_poem(begin_word, word_int_map, vocabularies, rnn_model))
