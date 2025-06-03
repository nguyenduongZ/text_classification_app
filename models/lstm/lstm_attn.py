import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, dropout=0.5):
        super(AttentionModel, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)

        self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.label = nn.Linear(hidden_size * 2, output_size)

    def attention_net(self, lstm_output, final_state):
        # lstm_output: (batch_size, seq_len, hidden_size * 2)
        # final_state: (2, batch_size, hidden_size)

        hidden = torch.cat((final_state[-2], final_state[-1]), dim=1).unsqueeze(2)  # (batch_size, hidden_size*2, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # (batch_size, seq_len)
        soft_attn_weights = F.softmax(attn_weights, dim=1)  # (batch_size, seq_len)

        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, input_sentences, batch_size=None):
        input = self.word_embeddings(input_sentences)
        input = input.permute(1, 0, 2)

        actual_batch_size = input.size(1) if batch_size is None else batch_size
        device = input.device

        h_0 = torch.zeros(2, actual_batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(2, actual_batch_size, self.hidden_size).to(device)

        output, (final_hidden_state, _) = self.lstm(input, (h_0, c_0))
        output = output.permute(1, 0, 2)

        attn_output = self.attention_net(output, final_hidden_state)
        logits = self.label(self.dropout(attn_output))

        return logits
