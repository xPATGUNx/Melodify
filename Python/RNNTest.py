import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import Python.MidiAndCSVTools as midi_tool

with open('../data/TrainData.csv', 'r') as data_reader:
    text = data_reader.read()

chars = tuple(set(text))

int2char = dict(enumerate(chars))

char2int = {ch: ii for ii, ch in int2char.items()}

encoded = np.array([char2int[ch] for ch in text])


def one_hot_encode(arr, n_labels):

    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_batches(arr, n_seqs, n_steps):
    """Create a generator that returns batches of size
       n_seqs x n_steps from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    """

    batch_size = n_seqs * n_steps
    n_batches = len(arr) // batch_size

    arr = arr[:n_batches * batch_size]

    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):

        x = arr[:, n:n + n_steps]

        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


class CharRNN(nn.Module):

    def __init__(self, tokens, n_steps=100, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # Define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, len(self.chars))

        self.init_weights()

    def forward(self, x, hc):
        """ Forward pass through the network.
            These inputs are x, and the hidden/cell state `hc`. """

        x, (h, c) = self.lstm(x, hc)

        x = self.dropout(x)

        x = x.reshape(x.size()[0] * x.size()[1], self.n_hidden)

        x = self.fc(x)

        return x, (h, c)

    def predict(self, char, h=None, cuda=False, top_k=None):
        """ Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        """
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        x = one_hot_encode(x, len(self.chars))

        inputs = torch.from_numpy(x)

        if cuda:
            inputs = inputs.cuda()

        h = tuple([each.data for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data

        if cuda:
            p = p.cpu()

        if top_k is None:
            top_ch = np.arange(len(self.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()

        char = np.random.choice(top_ch, p=p / p.sum())

        return self.int2char[char], h

    def init_weights(self):
        """ Initialize weights for fully connected layer """
        initrange = 0.1

        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs):
        """ Initializes hidden state """

        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, n_seqs, self.n_hidden).zero_(),
                weight.new(self.n_layers, n_seqs, self.n_hidden).zero_())


def train(network, data, epochs=10, n_seqs=10, n_steps=50, lr=0.001, clip=5, val_frac=0.1, cuda=False, print_every=10):
    """ Training a network

        Arguments
        ---------

        network: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        n_seqs: Number of mini-sequences per mini-batch, aka batch size
        n_steps: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        cuda: Train with CUDA on a GPU
        print_every: Number of steps for printing training and validation loss

    """

    network.train()

    opt = torch.optim.Adam(network.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if cuda:
        network.cuda()

    counter = 0
    n_chars = len(network.chars)

    for e in range(epochs):

        h = network.init_hidden(n_seqs)

        for x, y in get_batches(data, n_seqs, n_steps):

            counter += 1

            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            h = tuple([each.data for each in h])

            network.zero_grad()

            output, h = network.forward(inputs, h)

            loss = criterion(output, targets.view(n_seqs * n_steps).type(torch.cuda.LongTensor))

            loss.backward()

            nn.utils.clip_grad_norm_(network.parameters(), clip)

            opt.step()

            if counter % print_every == 0:

                # Get validation loss
                val_h = network.init_hidden(n_seqs)
                val_losses = []

                for x, y in get_batches(val_data, n_seqs, n_steps):

                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = network.forward(inputs, val_h)
                    val_loss = criterion(output, targets.view(n_seqs * n_steps).type(torch.cuda.LongTensor))

                    val_losses.append(val_loss.item())

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


def sample(network, size, prime='The', top_k=None, cuda=False):
    if cuda:
        network.cuda()
    else:
        network.cpu()

    network.eval()

    chars = [ch for ch in prime]

    h = network.init_hidden(1)

    for ch in prime:
        char, h = network.predict(ch, h, cuda=cuda, top_k=top_k)

    chars.append(char)

    for ii in range(size):
        char, h = network.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


if __name__ == '__main__':

    train_network = False

    if train_network:

        net = CharRNN(chars, n_hidden=512, n_layers=2)

        # print(net)

        n_seqs, n_steps = 256, 100

        train(net, encoded, epochs=25, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=True, print_every=10)

        model_name = 'rnn_1_epoch.net'

        checkpoint = {'n_hidden': net.n_hidden,
                      'n_layers': net.n_layers,
                      'state_dict': net.state_dict(),
                      'tokens': net.chars}

        with open(model_name, 'wb') as torch_writer:
            torch.save(checkpoint, torch_writer)

        print(sample(net, 2000, prime='1, 0, Note_on_c, 0, 57, 64', top_k=5, cuda=True))

    else:
        with open('rnn_1_epoch.net', 'rb') as neural_network:
            checkpoint = torch.load(neural_network)

        loaded = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
        loaded.load_state_dict(checkpoint['state_dict'])
        rnn_output = sample(loaded, 2000, cuda=True, top_k=5, prime='1, 0, Note_on_c, 0, 55, 72')

        # TODO: Implement Midi file creation from sample
        output_path = '../Generated Midi/generated.mid'
        midi_tool.generate_midi_from_output(rnn_output, output_path)
