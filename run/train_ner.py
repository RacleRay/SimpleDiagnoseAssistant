import json
import sys
import os
import tqdm

# 项目根文件夹 绝对路径
PROJECT_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import torch
import torch.optim as optim

# from tqdm import tqdm
from model.ner.bilstm_crf import BiLSTM_CRF
from model.ner.evaluate import evaluate
from util.tools import get_logger
from util.seq_process import sequence_mask_by_len


"""
首先说明，这种demo级别的模型，计算加上 NE Check 模型，输出的识别效果也很不好。计算有 0.8 的训练集f1
实际在 test 数据上的效果都挺感人的。
所以这个小模型，看看就好
"""

char_to_id = json.load(open('./data/char_to_id.json', mode='r', encoding='utf-8'))
train_data_file_path = "data/train_data.txt"
valid_data_file_path = "data/validate_data.txt"

EMBEDDING_DIM = 160
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.2

torch.manual_seed(23)
logger = get_logger()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, char_to_id, tag_to_ix):
        self.data_list = []
        for line in open(data_file, 'r', encoding='utf-8'):
            data = json.loads(line)
            self.data_list.append(data)

        self.char_to_id = char_to_id
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        sentence = data["text"]
        tags = data.get("label", None)

        length = len(sentence)
        length = torch.tensor(length, dtype=torch.long)

        char_ids = []
        default = char_to_id["UNK"]
        for ch in sentence:
            char_ids.append(self.char_to_id.get(ch, default))
        length = len(char_ids)
        length = torch.tensor(length, dtype=torch.long)
        char_ids = torch.tensor(char_ids, dtype=torch.long)

        if tags is not None:
            golden = torch.tensor([self.tag_to_ix[t] for t in tags],
                                  dtype=torch.long)
        else:
            golden = None

        return {"char_ids": char_ids,
                "golden": golden,
                "length": length}


def train(save_path):
    model = BiLSTM_CRF(len(char_to_id), EMBEDDING_DIM, HIDDEN_DIM).cuda()

    tag_to_ix = model.tag_to_ix

    id_to_char = {v: k for k, v in char_to_id.items()}
    id_to_tag = {v: k for k, v in tag_to_ix.items()}


    train_set = MyDataset(train_data_file_path, char_to_id, model.tag_to_ix)
    valid_set = MyDataset(valid_data_file_path, char_to_id, model.tag_to_ix)

    # 生成batch
    def collate_fn(samples):
        id_seqs = []
        tag_seqs = []
        length = []
        for sample in samples:
            id_seqs.append(sample["char_ids"])
            tag_seqs.append(sample["golden"])
            length.append(sample["length"].reshape(1))
        char_batch = torch.nn.utils.rnn.pad_sequence(id_seqs,
                                                    batch_first=True)
        tag_batch = torch.nn.utils.rnn.pad_sequence(tag_seqs,
                                                    batch_first=True,
                                                    padding_value=5)
        len_batch = torch.cat(length, dim=0)
        mask = sequence_mask_by_len(len_batch)
        return char_batch, tag_batch, len_batch, mask

    train_loader = torch.utils.data.DataLoader(train_set,
                                               BATCH_SIZE,
                                               shuffle=True,
                                               collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               BATCH_SIZE,
                                               shuffle=False,
                                               collate_fn=collate_fn)


    # train
    optimizer = optim.RMSprop(model.parameters(), LR)

    best_valid_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0.0
        cur_loss = 0.0
        valid_loss = 0.0

        model.train()
        for idx, batch in enumerate(train_loader):
            inputs, targets, length, mask = batch

            optimizer.zero_grad()

            loss = model.compute_logloss(inputs.cuda(), targets.cuda(), length.cpu(), mask.cuda())
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            cur_loss += loss.data.item()

            if (idx + 1) % 10 == 0:
                logger.info(f"step {idx+1}, train loss = {cur_loss / 10}")
                cur_loss = 0.0

        train_loss /= len(train_loader)
        logger.info(f"Epoch {epoch}, total_loss = {train_loss}")


        model.eval()
        tp, pred, gold = 0, 0, 0
        for idx, batch in enumerate(valid_loader):
            inputs, targets, length, mask = batch

            loss = model.compute_logloss(inputs.cuda(), targets.cuda(), length.cpu(), mask.cuda())
            valid_loss += loss.data.item()

            best_path = model(inputs.cuda(), length.cpu())

            # step
            acc, recall, f1_score, tp_count, pred_count, gold_count = evaluate(inputs,
                                                                              targets,
                                                                              best_path,
                                                                              id_to_char,
                                                                              id_to_tag)
            tp += tp_count
            pred += pred_count
            gold += gold_count

        valid_loss /= len(valid_loader)
        logger.info(f"Epoch {epoch}, valid_loss = {valid_loss}")

        if pred_count > 0:
            acc = tp / pred
            recall = tp / gold
            f1 = 2 * acc * recall / (acc + recall)
            logger.info(f"Epoch {epoch}, acc = {acc}, recall = {recall}, f1 = {f1}")

            if f1 > best_valid_f1:
                best_valid_f1 = f1
                torch.save(model.state_dict(), save_path)


def prepare_sequence(seq, char_to_id, device="cuda:0"):
    char_ids = []
    for idx, ch in enumerate(seq):
        if char_to_id.get(ch):
            char_ids.append(char_to_id[ch])
        else:
            char_ids.append(char_to_id["UNK"])
    return torch.tensor(char_ids, dtype=torch.long).unsqueeze(0).to(device), \
            torch.tensor(len(char_ids), dtype=torch.long).unsqueeze(0).cpu()


def predict(data_path, model_path, output_path):
    model = BiLSTM_CRF(len(char_to_id), EMBEDDING_DIM, HIDDEN_DIM).cuda()
    model.load_state_dict(torch.load(model_path))

    tag_to_ix = model.tag_to_ix
    id_to_tag = {v: k for k, v in tag_to_ix.items()}

    for fn in tqdm.tqdm(os.listdir(data_path)):
        fullpath = os.path.join(data_path, fn)
        entities_file = open(os.path.join(output_path, fn.replace('txt', 'csv')),
                             mode='w', encoding='utf-8')
        with open(fullpath, mode='r', encoding='utf-8') as f:
            content = f.readline()
            entities = []

            with torch.no_grad():
                sentence_in, length = prepare_sequence(content, char_to_id)
                best_path_list = model(sentence_in, length)
                best_path_list = best_path_list.reshape(-1)
                entity = None
                for char_idx, tag_id in enumerate(best_path_list):
                    tag_id = tag_id.cpu().item()
                    if tag_id in id_to_tag:
                        # 取字典的第一个字符
                        tag_index = id_to_tag[tag_id][0]
                        current_char = content[char_idx]
                        if tag_index == "B":
                            entity = current_char
                        elif tag_index == "I" and entity:
                            entity += current_char
                    if tag_id == tag_to_ix["O"] and entity:
                        # 为了防止特殊字符的干扰
                        if "、" not in entity and "～" not in entity and "。" not in entity \
                            and "”" not in entity and "：" not in entity and ":" not in entity \
                            and "，" not in entity and "," not in entity and "." not in entity \
                            and ";" not in entity and "；" not in entity and "【" not in entity \
                            and "】" not in entity and "[" not in entity and "]" not in entity:
                            entities.append(entity)
                        entity = None

        entities_file.write("\n".join(set(entities)))

    logger.info(f"预测完成，文件保存在 {output_path}")



if __name__ == '__main__':
    # train("weights/NER.bin")

    # predict
    # 对收集到的非结构化数据进行处理
    # predict(data_path="data/unstructured/", model_path="weights/NER.bin", output_path="data/ner_res/")