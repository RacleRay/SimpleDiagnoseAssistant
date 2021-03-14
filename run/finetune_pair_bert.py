import pandas as pd
from sklearn.utils import shuffle
from functools import reduce
from collections import Counter
import tqdm
import sys

# 项目根文件夹 绝对路径
PROJECT_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import torch
import torch.nn as nn
import torch.optim as optim

from model.sentpair.match import BERTPairFine, tokenizer
from util.tools import get_logger


# 若使用BERTPairFine为 bert whitening 预训练，注意在bert whitening初始化时，和此处参数保持一致
# ！！！max_len在使用 bert whitening 时，最好能够设置为最长文本的长度。这样，bert whitening的准确率会有较大提升。
embedding_size = 768
max_len = 20
fc_size = 16
epochs = 10
batch_size = 64
train_bert = True

lr = {"other": 0.005, "bert": 1e-5}

logger = get_logger()

"""
与 BERT 参数一起fine tune，交叉验证准确率达到 99%。任务简单，容易拟合甚至过拟合。需要看实际使用效果微调。
准确率达标的情况下，其实也不用再加上bertwhitening了。当任务要求提高，拟合效果变差时，可以尝试bertwhitening。
"""


def token_batch(text1, text2, max_len):
    # bert tokenize
    token_ids = tokenizer.encode(text1, text2)

    # 分隔符
    k = token_ids.index(102)

    if len(token_ids[:k]) >= max_len:
        token_ids_1 = token_ids[:max_len]
        masks_1 = [1] * max_len
    else:
        token_ids_1 = token_ids[:k] + (max_len - len(token_ids[:k])) * [0]
        masks_1 = [1] * (len(token_ids[:k])) + (max_len - len(token_ids[:k])) * [0]

    if len(token_ids[k:]) >= max_len:
        token_ids_2 = token_ids[k: k+max_len]
        masks_2 = [1] * max_len
    else:
        token_ids_2 = token_ids[k:] + (max_len - len(token_ids[k:])) * [0]
        masks_2 = [1] * len(token_ids[k:]) + (max_len - len(token_ids[k:])) * [0]

    token_ids = token_ids_1 + token_ids_2
    segments_ids = [0] * max_len + [1] * max_len
    masks = masks_1 + masks_2

    tokens_tensor = torch.tensor([token_ids])
    segments_tensor = torch.tensor([segments_ids])
    masks_tensor = torch.tensor([masks])

    return tokens_tensor, segments_tensor, masks_tensor


def data_loader(data_frame, split=0.1):
    "返回数据生成器"
    data = shuffle(data_frame).reset_index(drop=True)

    split_point = int(data.shape[0] * split)
    valid_data = data[:split_point]
    train_data = data[split_point:]

    if len(valid_data) < batch_size:
        raise("Batch size or split not match!")

    def _loader_generator(data):
        for batch in range(0, len(data), batch_size):
            batch_ids = []
            batch_type_ids = []
            batch_mask = []
            batch_labels = []
            max_len_half = max_len // 2
            for item in data[batch: batch + batch_size].values.tolist():
                # bert tokenize
                tokens_tensor, segments_tensor, masks_tensor = token_batch(item[1], item[2], max_len_half)

                # print(tokens_tensor)
                # print(segments_tensor)
                # print(masks_tensor)

                batch_ids.append(tokens_tensor)
                batch_type_ids.append(segments_tensor)
                batch_mask.append(masks_tensor)

                batch_labels.append([item[0]])

            ids = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_ids)
            types = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_type_ids)
            masks = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_mask)
            labels = torch.tensor(reduce(lambda x, y: x + y, batch_labels))

            yield (ids.cuda(), types.cuda(), masks.cuda(), labels.cuda())

    return _loader_generator(train_data), _loader_generator(valid_data), len(train_data), len(valid_data)



def train(data_path):
    data = pd.read_csv(data_path, header=None, sep="\t")

    logger.info("数据集的正负样本统计:")
    logger.info(dict(Counter(data[0].values)))

    model = BERTPairFine(max_len, embedding_size, fc_size, train_bert=train_bert).cuda()
    criterion = nn.CrossEntropyLoss()

    # 设置学习率
    if train_bert:
        params_lr = []
        for key, value in model.params.items():
            if key in lr:
                params_lr.append({"params": value, 'lr': lr[key]})
        optimizer = optim.AdamW(params_lr)
    else:
        optimizer = optim.Adam(model.parameters(), 0.02)

    best_valid_loss = 10000.
    for epoch in range(epochs):
        train_data_labels, valid_data_labels, train_data_len, valid_data_len = data_loader(data)

        train_running_loss = 0.0
        train_running_acc = 0.0
        model.train()
        for batch in tqdm.tqdm(train_data_labels):
            ids, types, masks, train_labels = batch
            optimizer.zero_grad()

            train_outputs = model(ids, types, masks)
            train_loss = criterion(train_outputs, train_labels)

            train_loss.backward()
            optimizer.step()

            train_running_loss += train_loss.item()
            train_running_acc += (train_outputs.argmax(1) == train_labels).sum().item()

        train_average_loss = train_running_loss * batch_size / train_data_len
        train_average_acc = train_running_acc / train_data_len

        logger.info(f"Epoch {epoch}   Train Loss: {train_average_loss}  |  Train Acc: {train_average_acc}")

        # valid
        valid_running_loss = 0.0
        valid_running_acc = 0
        model.eval()
        for batch in tqdm.tqdm(valid_data_labels):
            ids, types, masks, valid_labels = batch
            with torch.no_grad():
                valid_outputs = model(ids, types, masks)
                valid_loss = criterion(valid_outputs, valid_labels)

                valid_running_loss += valid_loss.item()
                valid_running_acc += (valid_outputs.argmax(1) == valid_labels).sum().item()

        valid_average_loss = valid_running_loss * batch_size / valid_data_len
        valid_average_acc = valid_running_acc / valid_data_len

        logger.info(f"Epoch {epoch}   Valid Loss:  {valid_average_loss}  |  Valid Acc: {valid_average_acc}")

        if valid_average_loss < best_valid_loss:
            best_valid_loss = valid_average_loss
            if not train_bert:
                torch.save(model.state_dict(), "weights/BERT_fine.bin")
            else:
                torch.save(model.state_dict(), "weights/BERT_fine_all.bin")


if __name__ == "__main__":
    train("data/sent_pair/train_data.csv")