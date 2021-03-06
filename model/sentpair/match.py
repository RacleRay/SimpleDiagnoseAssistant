import os
import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer, BertModel, BertConfig


MODEL_PATH = "d:\\Code\\Deep_Learning\\Project\\NLP_Project\\Doctor\\weights\\"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)


class BERTPairFine(nn.Module):
    """
    直接将这个模型用于预测匹配对，当做二分类问题，训练集准确率是很高的，这种场景不需要更复杂的模型就可以解决；
    但是更复杂一些的任务，给出一对句子, 使用1~5的评分评价两者在语义上的相似程度(STS-B)。
    没有标注，无监督任务时，可以使用BertWhitening。BERT在任务相关的语料中 fine tune 再使用。
    """
    def __init__(self, max_len=60, embed_size=768, fc_size=16, train_bert=False):
        super(BERTPairFine, self).__init__()
        self.train_bert = train_bert
        self.max_len = max_len
        self.embed_size = embed_size
        self.fc_size = fc_size
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embed_size * max_len, fc_size)
        self.fc2 = nn.Linear(fc_size, 2)

        config = BertConfig.from_json_file(os.path.join(MODEL_PATH, "config.json"))
        config.output_hidden_states = True

        self.model = BertModel.from_pretrained(MODEL_PATH, config=config).cuda()

        self.params = {"other": [], "bert": []}
        self.params["other"].extend([p for p in self.fc1.parameters()])
        self.params["other"].extend([p for p in self.fc2.parameters()])
        self.params["bert"].extend([p for p in self.model.parameters()])

    def forward(self, ids, types, masks):
        bert_out, _, _ = self.get_bert_encode(ids, types, masks)
        bert_out = bert_out.view(-1, self.embed_size * self.max_len)

        self.dropout(bert_out)
        out = F.relu(self.fc1(bert_out))
        # out = out.view(-1, self.max_len * self.fc_size)

        self.dropout(out)
        out = F.relu(self.fc2(out))

        return out

    def get_bert_encode(self, ids, types=None, masks=None):
        if not self.train_bert:
            with torch.no_grad():
                last_hidden_state, pooler_output, all_layerout = self.model(ids,
                                                                token_type_ids=types,
                                                                attention_mask=masks)
        else:
            last_hidden_state, pooler_output, all_layerout = self.model(ids,
                                                            token_type_ids=types,
                                                            attention_mask=masks)
        return last_hidden_state, pooler_output, all_layerout



class BertPairSim(nn.Module):
    """
    给出一对句子, 使用1~5的评分评价两者在语义上的相似程度(STS-B)，同时使用标注训练。
    针对 cosine similarity 评价指标，设计损失函数直接训练。STS-B任务 pearson correlation 0.83
    指标：
        Spear-man’s  rank  correlation  between  the  cosine  similarity  of  sentence  embeddings  and  the  gold  labels  on  multipledatasets.
    """
    def __init__(self, bert_size=768, hidden_size=1024, train_bert=False, **kwargs):
        super().__init__()
        self.train_bert = False

        config = BertConfig.from_json_file(os.path.join(MODEL_PATH, "config.json"))
        config.output_hidden_states = True

        self.model = BertModel.from_pretrained(MODEL_PATH, config=config).cuda()

        self.fc = nn.Linear(bert_size, hidden_size)
        self.activation = nn.Tanh()

        self.fc.weight.data.normal_(mean=0.0, std=0.02)

    def compute_cosim(self, vec1, vec2):
        return torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, ids, masks):
        if not self.train_bert:
            with torch.no_grad():
                last_hidden_state, pooler_output, all_layerout = self.model(ids,
                                                                attention_mask=masks)
        else:
            last_hidden_state, pooler_output, all_layerout = self.model(ids,
                                                            attention_mask=masks)

        outvec = self.activation(self.fc(pooler_output))
        return outvec

    def loss_fn(self, vec1, vec2, label, device):
        """
        vec1, vec2 分别为 ids1， ids2 通过 forward 计算的结果
            ```
            vec1 = model(ids1, mask)
            vec2 = model(ids2, mask)
            ```
        """
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
        l1_loss = torch.nn.L1Loss(reduction="mean")

        sim = cos(vec1, vec2)
        loss = l1_loss(sim, label)

        return loss


class BertWhitening(nn.Module):
    "无监督STS-B任务 0.72"
    def __init__(self, pretrained_path=None, **kwargs):
        super().__init__()
        self.pool = GlobalPooling1d(1)

        if not pretrained_path:
            config = BertConfig.from_json_file(os.path.join(MODEL_PATH, "config.json"))
            config.output_hidden_states = True
            self.use_pair_model = False
            self.model = BertModel.from_pretrained(MODEL_PATH, config=config).cuda()
        else:
            self.use_pair_model = True
            self.model = BERTPairFine(**kwargs)
            self.model.load_state_dict(torch.load(pretrained_path), strict=False)

    def forward(self, ids, mask=None, k_layers=2):
        vec = self.get_bert_embedding(ids, mask, k_layers)
        return vec

    def get_bert_embedding(self, ids, mask=None, k_layers=2):
        with torch.no_grad():
            if not self.use_pair_model:
                last_hidden_state, pooler_output, all_layerout = self.model(ids,
                                                                            attention_mask=mask)
            else:
                last_hidden_state, pooler_output, all_layerout = self.model.get_bert_encode(ids,
                                                                                            masks=mask)

        output = []
        for i in range(1, k_layers+1):
            output.append(self.pool(all_layerout[-i].transpose(1, 2),
                                    mask=mask))
        output = torch.cat(output, dim=-1).mean(dim=-1)

        # batch_size, time
        return output

    @staticmethod
    def compute_kernel_bias(all_vec):
        "计算输出阶段所用的参数，需要输入整理语料计算出的vector。BERT模型需要预训练。"
        vecs = np.concatenate(all_vec, axis=0)
        mu = vecs.mean(axis=0, keepdims=True)  # (1, 768)
        cov = np.cov(vecs.T)  # (768, 768)
        u, s, vh = np.linalg.svd(cov)  # (768, 768); (768,); (768, 768)
        W = np.dot(u, np.diag(1 / np.sqrt(s)))  # (768, 768)
        return W[:, :256], -mu  # 256的主成分

    @staticmethod
    def transform_and_normalize(vecs, kernel=None, bias=None):
        "应用kernel 和 bias变换，然后标准化"
        if not (kernel is None or bias is None):
            vecs = (vecs + bias).dot(kernel)  # (#vecs, 256)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5

    def metric(self, vecs_1, vecs_2, kernel, bias, labels):
        "计算相关系数"
        trans_1 = self.transform_and_normalize(vecs_1, kernel, bias)
        trans_2 = self.transform_and_normalize(vecs_2, kernel, bias)
        sim = (trans_1 * trans_2).sum(axis=1)
        corroef = self.compute_corrcoef(labels, sim)
        return corroef

    @staticmethod
    def compute_corrcoef(x, y):
        "Spearman相关系数"
        return scipy.stats.spearmanr(x, y).correlation

    def predict(self, vecs_1, vecs_2, kernel, bias):
        "计算余弦相似度"
        trans_1 = self.transform_and_normalize(vecs_1, kernel, bias)
        trans_2 = self.transform_and_normalize(vecs_2, kernel, bias)
        sim = (trans_1 * trans_2).sum(axis=1)

        # trans_1 = trans_1.reshape(-1)
        # trans_2 = trans_2.reshape(-1)

        return sim



class GlobalPooling1d(nn.AdaptiveAvgPool1d):
    def __init__(self, output_size):
        super(GlobalPooling1d, self).__init__(output_size)

    def forward(self, inputs, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            inputs = inputs * mask
            return F.adaptive_avg_pool1d(inputs, self.output_size)
        else:
            return F.adaptive_avg_pool1d(inputs, self.output_size)


