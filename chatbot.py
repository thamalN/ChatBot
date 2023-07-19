import json
import random
import torch

from model import NeuralNet
from utils import tokenize, bag_of_words


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
vocab = data["vocab"]
tags = data["tags"]

model = NeuralNet(input_size=input_size, hidden_layers=hidden_size, output_size=output_size).to(device)
model.load_state_dict(model_state)

model.eval()

bot_name = "Shura"
print("type 'quit' to exit")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, vocab)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        output = model(X)

    _, predicted_idx = torch.max(output, dim=1)
    tag = tags[predicted_idx.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted_idx.item()]

    if prob.item() >= 0.8:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f'{bot_name}: {random.choice(intent["responses"])}')
    else:
        print(f'{bot_name}: I do not understand')

