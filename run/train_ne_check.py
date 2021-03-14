import os
import sys

# 项目根文件夹 绝对路径
PROJECT_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from model.encoder.bert import get_bert_embedding
from model.neCheck.rnn import NEmodel
from sklearn.model_selection import train_test_split
from util.tools import get_logger


hidden_size = 128
input_size = 768
n_categories = 2
learning_rate = 0.005

epochs = 10
batch_size = 16


logger = get_logger()


class MyDatasset(torch.utils.data.Dataset):
    def __init__(self, data, max_len=24):
        self.data = data.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x, y = self.data.loc[idx, "text"], self.data.loc[idx, 'label']
        length = len(x)
        length = torch.tensor(length, dtype=torch.long)

        embed = get_bert_embedding(x)
        embed = torch.cat((embed, torch.zeros(embed.shape[0], self.max_len - embed.shape[1], embed.shape[2]).cuda()), dim=1)
        embed = embed.squeeze(0)

        label = torch.tensor(y, dtype=torch.long)
        return embed, label, length


def get_train_example(train_data_path):
    raw = pd.read_csv(train_data_path, names=['label', 'text'], encoding='utf-8',
                      dtype={'label': np.int, 'text': str})
    train, valid = train_test_split(raw, test_size=0.2, random_state=42)

    train_set = MyDatasset(train)
    valid_set = MyDatasset(valid)

    return train_set, valid_set


def train(data_path, save_path):
    train_set, valid_set = get_train_example(data_path)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model = NEmodel(input_size, hidden_size, n_categories).cuda()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    best_valid_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        cur_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        model.train()
        for idx, batch in enumerate(train_dataloader):
            embed, label, length = batch

            optimizer.zero_grad()
            prediction = model(embed, length)

            loss = criterion(prediction, label.cuda())
            loss.backward()

            optimizer.step()

            train_loss += loss.data.item()
            cur_loss += loss.data.item() * batch_size
            train_acc += (prediction.argmax(1) == label.cuda()).sum().item()
            if (idx + 1) % 100 == 0:
                train_acc /= 100 * batch_size
                logger.info(f"step {idx+1}, train_acc = {train_acc}, loss = {cur_loss / 100}")

                train_acc = 0.0
                cur_loss = 0.0

        train_loss /= len(train_dataloader)
        logger.info(f"Epoch {epoch}, total_loss = {train_loss}")


        model.eval()
        for idx, batch in enumerate(valid_dataloader):
            embed, label, length = batch
            prediction = model(embed, length)
            loss = criterion(prediction, label.cuda())

            valid_loss += loss.data.item() * batch_size
            valid_acc += (prediction.argmax(1) == label.cuda()).sum().item()

        valid_loss /= len(valid_dataloader)
        valid_acc /= (len(valid_dataloader) * batch_size)

        logger.info(f"Epoch {epoch}, valid total_loss = {train_loss}, valid acc = {valid_acc}")

        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), save_path)



def predict(input_path, output_path, model_path):
    "对于没有审核的数据，使用模型进行审核，这部分要保证在足够多的审核数据中训练条件下，才能使用"
    files = os.listdir(input_path)

    model = NEmodel(input_size, hidden_size, n_categories).cuda()
    model.load_state_dict(torch.load(model_path))

    for file in tqdm.tqdm(files):
        with open(os.path.join(output_path, file), "w", encoding="utf-8") as fout:
            with open(os.path.join(input_path, file), "r", encoding="utf-8") as fin:
                line = fin.readline()
                while len(line) > 0:
                    line = line.strip()
                    embed = get_bert_embedding(line)
                    # 有英文字符，取shape不会出bug
                    length = torch.tensor([embed.shape[1]], dtype=torch.long)

                    with torch.no_grad():
                        prediction = model(embed, length)

                    _, top_index = prediction.topk(1, 1)

                    if top_index.cpu().item() == 1:
                        fout.write(line + '\n')

                    line = fin.readline()


if __name__ == "__main__":
    # 0.9878
    # train("./data/structured/necheck_train.csv", "./weights/NE_CHECK.bin")

    predict("data/structured/noreview", "data/structured/reviewed", "./weights/NE_CHECK.bin")