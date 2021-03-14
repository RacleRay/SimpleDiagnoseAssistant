import torch
import torch.nn as nn


class NEmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, length):
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs,
                                                         length.to('cpu'),
                                                         batch_first=True,
                                                         enforce_sorted=False)
        step_outs, hn = self.rnn(inputs)

        step_outs, len_unpack = torch.nn.utils.rnn.pad_packed_sequence(step_outs, batch_first=True)

        hn = hn.squeeze(0)
        out = self.fc(hn)
        out = self.softmax(out)
        return out



class RNN(nn.Module):
    "Naive rnn implementarion"
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.fc_1 = nn.Linear(input_size + hidden_size, hidden_size)
        self.fc_2 = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, hidden, length):
        hidden = torch.cat([inputs, hidden], 1)
        hidden = self.fc_1(hidden)
        output = self.fc_2(hidden)
        output = self.softmax(output)
        return output, hidden