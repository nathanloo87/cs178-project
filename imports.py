# im just gonna move all the imports here so it doesn't look shitty in the main files

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import SVHN
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import h5py
