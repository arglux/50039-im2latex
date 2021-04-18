from dataset import Vocabulary, LatexDataset, device
import models, random, time
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, criterion, optimizer, force_ratio):
    total_loss = 0
    total_items = 0

    for img, eqn in tqdm(loader):
        if eqn.shape[1] > 30: continue
        optimizer.zero_grad()
        if random.random() < force_ratio:
            outs = model(img, force_inp=eqn)
        else:
            outs = model(img, length=eqn.shape[1])

        loss = criterion(outs, eqn)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_items += 1

    return total_loss / len(loader)


def train_epochs(model, epochs, lr=1e-3, force_ratio=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        loss = train(model, criterion, optimizer, force_ratio)
        print(f"Loss {loss}")

        torch.save(
            model.state_dict(), "model" + time.strftime("%Y-%m-%d_%H-%M-%S") + ".pt"
        )


if __name__ == "__main__":
    vocab = Vocabulary()
    dataset = LatexDataset(vocab, "train")
    loader = DataLoader(dataset)

    hidden_size = 256
    model = models.Transcriptor(hidden_size, len(vocab)).to(device)

    train_epochs(model, 5)
