import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')
X = np.moveaxis(train_data['X'], -1, 0)
y = train_data['y'].flatten()
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])

X_hog = np.array([hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in X_gray])
