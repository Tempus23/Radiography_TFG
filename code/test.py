import os
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from data import loader_train, loader_val, loader_test
from train import train_model
from models import RadiographyClassifier

#train a model
model = RadiographyClassifier(5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_model(model, loader_train, loader_val, criterion, optimizer, device, num_epochs=10)