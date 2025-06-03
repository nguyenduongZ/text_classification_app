import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, weights, dropout=0.5):
        super(LSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)

        self.lstm = nn.LSTM(embedding_length, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.label = nn.Linear(hidden_size * 2, output_size)  # *2 vì bidirectional

    def forward(self, input_sentence, batch_size=None):
        input = self.word_embeddings(input_sentence)
        input = input.permute(1, 0, 2)

        actual_batch_size = input.size(1) if batch_size is None else batch_size
        device = input.device

        h_0 = torch.zeros(2, actual_batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(2, actual_batch_size, self.hidden_size).to(device)

        output, (final_hidden_state, _) = self.lstm(input, (h_0, c_0))

        final_hidden = torch.cat((final_hidden_state[-2], final_hidden_state[-1]), dim=1)
        final_output = self.label(self.dropout(final_hidden))

        return final_output
