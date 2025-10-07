import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import kagglehub

path = kagglehub.dataset_download("abhinavnayak/catsvdogs-transformed")

print(path)


#bilder sind schon auf 224x224 zugeschnitten

transform = transforms.Compose([transforms.ToTensor()])
data = datasets.ImageFolder(root=path, transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
print(len(data))