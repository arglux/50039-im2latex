import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Convolutional(nn.Module):
    def __init__(self, output_size):
        super(Convolutional, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# TODO Eventually change this to a bidirectional LSTM
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # TODO Consider adding an FC layer to convert input_size => hidden_size?
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def step(self, inp, hid):
        out, hid = self.gru(inp, hid)
        return out, hid

    def forward(self, inp):
        inp_length = inp.shape[0]
        outs = torch.zeros(inp_length, self.hidden_size, device=device)

        hid = self.init_hidden()
        for ei in range(inp_length):
            out, hid = self.step(inp[ei].view(1, 1, -1), hid)
            outs[ei] += out[0, 0]

        return outs

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.hidden_size = hidden_size

        # TODO make attention layer more sophisticated?
        self.attention = nn.Linear(self.hidden_size * 2, 1)
        self.contextualizer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), nn.ReLU()
        )

    def forward(self, inp, hid, V):
        h, w, _ = V.shape
        alpha = torch.zeros(h, w, device=device)
        for i in range(h):
            for j in range(w):
                alpha[i, j] = self.attention(torch.cat((hid[0, 0], V[i, j])))

        alpha = F.softmax(alpha.flatten(), dim=0).reshape((h, w))
        attn = torch.einsum("ij,ijk->k", alpha, V)

        ctx = self.contextualizer(torch.cat((inp[0, 0], attn)))

        return ctx.unsqueeze(0).unsqueeze(0)


class PredictiveAlignmentAttention(nn.Module):
    def __init__(self, hidden_size, sx=1, sy=1):
        super(PredictiveAlignmentAttention, self).__init__()
        self.hidden_size = hidden_size
        self.sx = sx
        self.sy = sy

        # TODO consider decomposing this into two separate layers
        # TODO consider factoring in V.mean() as an input
        self.alignment = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, 2),
            nn.Sigmoid(),
        )
        self.contextualizer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), nn.ReLU()
        )

    def forward(self, inp, hid, V):
        h, w, _ = V.shape

        algn = self.alignment(hid)
        px, py = algn.squeeze()
        x0 = px * w
        y0 = py * h

        def gaussian(x, y):
            return torch.exp(
                -(
                    (x - x0) ** 2 / (2 * self.sx ** 2)
                    + (y - y0) ** 2 / (2 * self.sy ** 2)
                )
            )

        xs = torch.cat(h * [torch.arange(w, device=device).unsqueeze(0)])
        ys = torch.cat(w * [torch.arange(h, device=device).unsqueeze(1)], axis=1)
        alpha = gaussian(xs, ys)

        alpha = F.softmax(alpha.flatten(), dim=0).reshape((h, w))
        attn = torch.einsum("ij,ijk->k", alpha, V)

        ctx = self.contextualizer(torch.cat((inp[0, 0], attn)))

        return ctx.unsqueeze(0).unsqueeze(0)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attention = PredictiveAlignmentAttention(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size), nn.LogSoftmax(dim=1)
        )

    def step(self, inp, hid, V):
        inp = self.embedding(inp).view(1, 1, -1)
        inp = self.dropout(inp)

        ctx = self.attention(inp, hid, V)

        out, hid = self.gru(ctx, hid)
        out = self.classifier(out[0])

        return out, hid

    def forward(self, V, inp=1, eos=0, force_inp=None, length=None):
        inp = torch.tensor(inp, device=device)
        hid = self.init_hidden()
        outs = []
        if force_inp != None:
            for inp in force_inp.squeeze():
                out, hid = self.step(inp, hid, V)
                outs.append(out)
        else:
            ind = 0
            while (not length) or (ind < length):
                out, hid = self.step(inp, hid, V)
                topv, topi = out.topk(1)
                outs.append(out)
                if (not length) and topi.item() == eos:
                    break
                inp = topi.squeeze().detach()
                ind += 1

        return torch.stack(outs).permute(1, 2, 0)

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Transcriptor(nn.Module):
    def __init__(self, hidden_size, output_size, fmap_size=64):
        super(Transcriptor, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fmap_size = fmap_size

        self.convolutional = Convolutional(self.fmap_size)
        self.encoder = EncoderRNN(self.fmap_size, self.hidden_size)
        self.decoder = AttnDecoderRNN(self.hidden_size, self.output_size)

    def forward(self, img, force_inp=None, length=None):
        fmap = self.convolutional(img)
        fmap = fmap.squeeze(dim=0).permute(1, 2, 0)
        V = torch.stack([self.encoder(row) for row in fmap])
        outs = self.decoder(V, force_inp=force_inp, length=length)
        return outs
