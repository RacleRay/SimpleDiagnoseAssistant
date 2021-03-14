import pandas as pd
import numpy as np
import sys
import pickle
import tqdm
import torch
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from functools import reduce

# 项目根文件夹 绝对路径
PROJECT_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from model.sentpair.match import BertWhitening, tokenizer


"""
BertWhitening无监督方法。
任务难度可以匹配像 无监督预测多个等级划分 这样的情况。

这个模型的设计目的就是针对 较难的句向量语义表示任务 提出的。

简单监督学习句子配对任务，用不着这个模型。

但是实际没有这么多监督数据时，可使用BertWhitening进行匹配、
"""


# ！！！ 关键参数，由于是整个句子的语义表示，最好能够覆盖全句。
# 若 使用BERTPairFine为 bert whitening 预训练, max_len 要保持一致。
# 这里完全可以 使用专门的 fine tune 模型，而不使用 BERTPairFine， 从而摆脱 max_len 的一致性限制。
# 也就是 BERT 在 cls 处，接 fc 层等。
# 但是，即使不 fine tune，bert whitening也能有较好的准确率。
max_len = 128


def train(data_path, batch_size=16, num_batches=4000, pretrained_path=None):
    "num_batches * batch_size超出总数据量时，以总数据量为准"
    if pretrained_path is None:
        # 使用完全没有在相关任务上fine tune的模型，看看效果
        model = BertWhitening().cuda()
    else:
        # pretrained_path：在相关任务上fine tune的模型保存路径, 参数保持一致
        model = BertWhitening(pretrained_path=pretrained_path,
                                              max_len=max_len,
                                              embed_size=768,
                                              fc_size=16, 
                                              train_bert=False).cuda()

    data, length = data_loader(data_path, batch_size, num_batches)

    print("Computing vecs ... ")
    vecs_list_1 = []
    vecs_list_2 = []
    labels_list = []
    for ids_1, ids_2, mask_1, mask_2, labels in tqdm.tqdm(data):
        vec_1 = model(ids_1, mask_1)
        vec_2 = model(ids_2, mask_2)
        vecs_list_1.append(vec_1.detach().cpu().numpy())
        vecs_list_2.append(vec_2.detach().cpu().numpy())
        labels_list.append(labels)

    print("Computing kernel and bais ...")
    kernel, bias = model.compute_kernel_bias(
        [v for vecs in [vecs_list_1, vecs_list_2] for v in vecs]
    )

    with open("weights/kernel.bin", "wb") as f:
        pickle.dump(kernel, f)

    with open("weights/bias.bin", "wb") as f:
        pickle.dump(bias, f)

    print("Kernel and bias saved in weights/")

    # 评测结果, 当数据量太大，每个batch计算,注意batch大一点，不然 label 全是一样的话，spearmanr 相关系数中的 variance 为 0，计算异常
    # corroef_record = []
    # for a_vecs, b_vecs, labels in zip(vecs_list_1, vecs_list_2, labels_list):
    #     corroef = model.metric(a_vecs, b_vecs, kernel, bias, labels)
    #     corroef_record.append(corroef)

    # 一起算
    corroef = model.metric(np.concatenate(vecs_list_1, axis=0),
                           np.concatenate(vecs_list_2, axis=0),
                           kernel,
                           bias,
                           np.array([la for labels in labels_list for la in labels]))
    print("皮尔逊相关系数", corroef)

    pred = model.predict(np.concatenate(vecs_list_1, axis=0),
                                    np.concatenate(vecs_list_2, axis=0),
                                    kernel, bias)

    pred = np.array(list(map(lambda x: 0 if x < 0 else 1, pred)))
    print("模型accuracy：", accuracy_score(np.array([la for labels in labels_list for la in labels]),
                                                                    pred))

    # 预测一组，查看效果
    pred = model.predict(np.stack(vec_1.detach().cpu().numpy(), axis=0),
                                      np.stack(vec_2.detach().cpu().numpy(), axis=0),
                                        kernel, bias)
    # 查看一组预测
    print(pred)
    print(labels)


def data_loader(data_path, batch_size, num_batches=1000):
    data = pd.read_csv(data_path, header=None, sep="\t")

    data = shuffle(data).reset_index(drop=True)

    def _loader_generator(data):
        count = 0
        for batch_start in range(0, len(data), batch_size):
            if count > num_batches:
                break
            count += 1
            batch_ids_1 = []
            batch_ids_2 = []
            batch_masks_1 = []
            batch_masks_2 = []
            batch_labels = []
            for item in data[batch_start: batch_start + batch_size].values.tolist():
                out_1 = tokenizer.encode_plus(item[1], max_length=max_len, pad_to_max_length=True,
                                            return_tensors='pt', return_attention_mask=True)
                out_2 = tokenizer.encode_plus(item[2], max_length=max_len, pad_to_max_length=True,
                                            return_tensors='pt', return_attention_mask=True)

                batch_ids_1.append(out_1["input_ids"])
                batch_ids_2.append(out_2["input_ids"])
                batch_masks_1.append(out_1["attention_mask"])
                batch_masks_2.append(out_2["attention_mask"])
                batch_labels.append([item[0]])

            ids_1 = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_ids_1)
            ids_2 = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_ids_2)
            mask_1 = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_masks_1)
            mask_2 = reduce(lambda x, y: torch.cat((x, y), dim=0), batch_masks_2)
            labels = reduce(lambda x, y: x + y, batch_labels)
            yield (ids_1.cuda(), ids_2.cuda(), mask_1.cuda(), mask_2.cuda(), labels)

    return _loader_generator(data), len(data)


if __name__=="__main__":
    train("data/sent_pair/train_data.csv", batch_size=32, num_batches=1000)

    # train("data/sent_pair/train_data.csv", batch_size=32, num_batches=1000, pretrained_path="weights/BERT_fine_all.bin")