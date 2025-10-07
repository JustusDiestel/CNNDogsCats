import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import kagglehub

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)



path = kagglehub.dataset_download("abhinavnayak/catsvdogs-transformed")

print(path)
print("Inhalt von:", os.path.join(path, "train_transformed"))
print(os.listdir(os.path.join(path, "train_transformed")))

#bilder sind schon auf 224x224 zugeschnitten

transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])

data = datasets.ImageFolder(root=os.path.join(path, "train_transformed"), transform=transform)

print(len(data))

indices = list(range(len(data)))
train_idx, test_idx = train_test_split(indices, train_size=0.8, shuffle=True)
train_data = Subset(data, train_idx)
test_data = Subset(data, test_idx)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    running_loss = 0.0
    for imgs, labels in train_dataloader:
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_dataloader):.4f}")