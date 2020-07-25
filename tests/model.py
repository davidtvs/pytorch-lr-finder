import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LinearMLP(nn.Module):
    def __init__(self, layer_dim):
        super(LinearMLP, self).__init__()
        io_pairs = zip(layer_dim[:-1], layer_dim[1:])
        layers = [nn.Linear(idim, odim) for idim, odim in io_pairs]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMTagger(nn.Module):
    """
    A POS (part-of-speech) tagger using LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, tagset_size, vocab_to_ix):
        """
        Arguments:
            embedding_dim (int): dimension of output embedding vector.
            hidden_dim (int): dimension of LSTM hidden layers.
            target_size (int): number of tags for tagger to learn.
            vocab_to_ix (dict): a dict for vocab to index conversion.
        """
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_to_ix = vocab_to_ix

        self.word_embeddings = nn.Embedding(len(vocab_to_ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def get_indices_of_vocabs(self, vocabs):
        return torch.LongTensor([self.vocab_to_ix[v] for v in vocabs])

    def forward(self, vocabs):
        """
        Arguments:
            vocabs (list of str): tokenized sentence.
        """
        device = next(self.lstm.parameters()).device

        # convert vocabs to indices and get embedding vectors
        indices = self.get_indices_of_vocabs(vocabs).to(device)
        embeds = self.word_embeddings(indices)

        lstm_out, _ = self.lstm(embeds.view(len(vocabs), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(vocabs), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
