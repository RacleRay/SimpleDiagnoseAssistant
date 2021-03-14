from flask import Flask
from flask import request
app = Flask(__name__)

import sys
# 项目根文件夹 绝对路径
PROJECT_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor"
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

import pickle
import torch
from model.sentpair.match import BertWhitening, BERTPairFine, tokenizer


embedding_size = 768
max_len = 20
fc_size = 16

enable_pairmodel = True
enable_whitemodel = False

if enable_pairmodel:
    MODEL_PATH = "weights/BERT_fine_all.bin"
    model = BERTPairFine(max_len, embedding_size, fc_size).cuda()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

if enable_whitemodel:
    KERNEL = "weights/kernel.bin"
    BIAS = "weights/bias.bin"

    model2 = BertWhitening().cuda()
    model2.eval()

    with open(KERNEL, 'rb') as f:
        kernel = pickle.load(f.read())

    with open(BIAS, 'rb') as f:
        bias = pickle.load(f.read())


@app.route('/v1/recognition/', methods=["POST"])
def recognition():
    "使用 BERTPairFine 预测"
    text_1 = request.form['text1']
    text_2 = request.form['text2']

    tokens_tensor, segments_tensor, masks_tensor = token_batch(text_1,
                                                               text_2,
                                                               max_len_each=10)
    outputs = model(tokens_tensor, segments_tensor, masks_tensor)
    _, predicted = torch.max(outputs, 1)

    return str(predicted.item())


@app.route('/v1/whitening/', methods=["POST"])
def whitening():
    "使用 BertWhitening 预测"
    max_len = 200

    text_1 = request.form['text1']
    text_2 = request.form['text2']

    out_1 = tokenizer.encode_plus(text_1,
                                  max_length=max_len,
                                  pad_to_max_length=True,
                                  return_tensors='pt',
                                  return_attention_mask=True)
    out_2 = tokenizer.encode_plus(text_2,
                                  max_length=max_len,
                                  pad_to_max_length=True,
                                  return_tensors='pt',
                                  return_attention_mask=True)

    vec_1 = model2(out_1["input_ids"].cuda(), out_1["attention_mask"].cuda())
    vec_2 = model2(out_2["input_ids"].cuda(), out_2["attention_mask"].cuda())

    sim = model.predict(vec_1.detach().cpu().numpy(),
                  vec_2.detach().cpu().numpy(), kernel, bias)

    pred = 0 if sim < 0 else 1

    return str(pred)


def token_batch(text1, text2, max_len_each=10):
    # bert tokenize
    token_ids = tokenizer.encode(text1, text2)

    # 分隔符
    k = token_ids.index(102)

    if len(token_ids[:k]) >= max_len_each:
        token_ids_1 = token_ids[:max_len_each]
        masks_1 = [1] * max_len_each
    else:
        token_ids_1 = token_ids[:k] + (max_len_each - len(token_ids[:k])) * [0]
        masks_1 = [1] * (len(
            token_ids[:k])) + (max_len_each - len(token_ids[:k])) * [0]

    if len(token_ids[k:]) >= max_len_each:
        token_ids_2 = token_ids[k:k + max_len_each]
        masks_2 = [1] * max_len_each
    else:
        token_ids_2 = token_ids[k:] + (max_len_each - len(token_ids[k:])) * [0]
        masks_2 = [1] * len(
            token_ids[k:]) + (max_len_each - len(token_ids[k:])) * [0]

    token_ids = token_ids_1 + token_ids_2
    segments_ids = [0] * max_len_each + [1] * max_len_each
    masks = masks_1 + masks_2

    tokens_tensor = torch.tensor([token_ids])
    segments_tensor = torch.tensor([segments_ids])
    masks_tensor = torch.tensor([masks])

    return tokens_tensor.cuda(), segments_tensor.cuda(), masks_tensor.cuda()

