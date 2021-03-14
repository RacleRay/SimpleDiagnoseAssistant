import torch
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor\\weights\\')
model = BertModel.from_pretrained('d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor\\weights\\').cuda()


def get_bert_embedding(inputs):
    "只需要bert embedding"
    inputs = tokenizer.encode(inputs, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state, pooler_output = model(inputs.cuda())
    return last_hidden_state[:, 1:-1, :]