import torch
import torch.nn as nn
from model.ner.crf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tag_to_ix=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        if tag_to_ix:
            self.tag_to_ix = tag_to_ix
        else:
            self.tag_to_ix = {"O":0, "B-dis":1, "I-dis":2, "B-sym":3, "I-sym":4, "TAGPAD": 5}
        self.tagset_size = len(self.tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.embed_drop = nn.Dropout(0,2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        # self.dropout = nn.Dropout(0.2)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.crf = CRF(self.tagset_size, batch_first=True)


    def compute_logloss(self, text, tags, length, mask=None):
        """计算loss，更新model 和 crf 转移矩阵参数

        https://racleray.github.io/2020/11/18/CRF-SimpleNote/"""

        lstm_out = self.bilstm(text, length)
        loss = self.crf(emissions=lstm_out, tags=tags, mask=mask)
        return -loss

    def forward(self, text, length, mask=None):
        "根据 model 的输出 和 crf 转移矩阵，viterbi解码，得到最优解"
        lstm_out = self.bilstm(text, length)
        best_path = self.crf.decode(lstm_out, mask, pad_tag=self.tag_to_ix["TAGPAD"])
        return best_path

    def bilstm(self, text, length):
        self.batch_size = text.size(0)

        embeds = self.word_embeds(text)
        embeds = self.embed_drop(embeds)

        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, length, batch_first=True, enforce_sorted=False)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out = self.dropout(lstm_out)

        # 从hidden_dim变换成转移矩阵的维度tagset_size
        lstm_out = self.hidden2tag(lstm_out)
        return lstm_out
