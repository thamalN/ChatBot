import json
import numpy as np
import torch
import torch.nn as nn

from model import NeuralNet
from utils import tokenize, stem, bag_of_words
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

vocab = []
tags = []
xy = []
ignore_words = ['?', '!', ',', '.']

for intent in intents["intents"]:
    tag = intent["tag"]
    tags.append(tag)

    for pattern in intent["patterns"]:
        words = tokenize(pattern)
        vocab.extend(words)
        xy.append((words, tag))

vocab = [stem(word) for word in vocab if word not in ignore_words]
vocab = sorted(set(vocab))
tags = sorted(set(tags))

x_train = []
y_train = []


for token_sentence, tag in xy:
    bag = bag_of_words(token_sentence, vocab)
    x_train.append(bag)
    # print(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = x_train
        self.y_data = y_train
        self.n_samples = x_train.shape[0]

    def __getitem__(self, x):
        return self.x_data[x], self.y_data[x]

    def __len__(self):
        return self.n_samples


batch_size = 8
learning_rate = 0.001
n_epochs = 1000
input_size = x_train.shape[1]
hidden_layers = 8
n_classes = len(tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=input_size, hidden_layers=hidden_layers, output_size=n_classes).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = []
losses = []


for epoch in range(n_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.int64)

        outputs = model(words)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch = {epoch + 1} / {n_epochs} | loss = {loss.item(): .3f}')

print(f'final loss = {loss.item():.3f}')

f.close()

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_layers,
    "output_size": n_classes,
    "vocab": vocab,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training Completed. File saved to {FILE}')
