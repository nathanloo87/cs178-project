import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.io import loadmat

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

X = np.moveaxis(train_data['X'], -1, 0)
y = train_data['y'].flatten()
X_gray = np.dot(X[..., :3], [0.2989, 0.5870, 0.1140])

X_hog = np.array([hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2)) for img in X_gray])

X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

accuracy = nb_model.score(X_test, y_test)
print(f"Naïve Bayes Accuracy on SVHN: {accuracy:.4f}")